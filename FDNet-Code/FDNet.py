from tkinter import W
from tkinter.tix import MAIN
from turtle import mainloop
from unicodedata import name
from pip import main

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import complexPyTorch.complexLayers as CPL

from mmseg.models.builder import BACKBONES



BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

from .ops_dcnv3.modules.my_dcnv3 import SDConv_stage



SIZE=448
# SIZE=512 

def complex_gelu(input):
    return F.gelu(input.real).type(torch.complex64)+1j*F.gelu(input.imag).type(torch.complex64)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W,stage):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            PatchMerging(in_channels, out_channels),
            FDConv_Block(out_channels, out_channels, resolution_H=H, resolution_W=W,stage=stage)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BAM(nn.Module):
    """
    中间平均池化、max池化 的结构，提取全局
    """
    def __init__(self, in_channels, W, H, freq_sel_method = 'top16'):
        super(BAM, self).__init__()
        self.in_channels = in_channels

        # local channel
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.tw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                            groups=self.in_channels)
        self.twln = nn.LayerNorm([self.in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))


    def forward(self, x):
        N, C, H, W = x.shape  # global
        # self and local
        # w fusion 权重融合的过程
        x_s = (self.wmax * self.maxpool(x).squeeze(-1)) + self.wdct * (self.gap(x).squeeze(-1))
        # attention weights
        x_s = x_s.unsqueeze(-1)
   
        att_c =self.sigmoid(self.twln(self.tw(x_s)))
        return att_c

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Linear(dim, dim * 4, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, H * 2, W * 2, C // 4)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


def sdconv_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    
    # 对input生成网格位置
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device) # 高宽作为分母，那自然offset是0-1的

    # print(f'offset {offset.shape} norm {spatial_norm.shape}')
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm # TODO 如何约束cross元素的移动
    
    
    
    P_ = kernel_h * kernel_w

    # 说明sample-location是【0,1】的 *2-1 会把offset 移到【-1,1】
    sampling_grids = 2 * sampling_locations - 1 # 计算采样的网络 则是对采样位置进行一个线性变换。乘以2会扩大位置范围，然后减去1会将位置平移到[-1, 1]范围内
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in) # 分组
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1) # 采样位置 b*group,h*w,p,2
    
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample( # 采样插值 对输入图像进行网格采样，即使用采样网格在输入图像上进行插值，以获得在卷积操作中所需的特征。
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_) # 分组的
    # 遮罩操作：将采样的结果与遮罩进行逐元素相乘。遮罩指示了哪些位置的特征应该参与计算，这一步类似于卷积中的权重操作。
    # 对采样 乘掩码 即卷积核权重，会有一个sum求和
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, H_out*W_out)
    # p维度求和 p=k_w*k_h 相当于卷积求和了，b,head*c/head, h*w--> b,h,w,c

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()



class SDConv_stage(nn.Module):
    """ 加入sd约束变形，kersize=5"""
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            
            group=16,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
   
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape
        # print(f'heere is hf ')

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)

            # TODO 对offset的cross 进行约束控制
        offset = offset.reshape(N,self.group,self.kernel_size,self.kernel_size,2,H,W) # b,g,kw,kh,2,h,w
        # offset_new = torch.zeros_like(offset) #0矩阵
        center = int(self.kernel_size//2)
        for index in range(1, center+1): # 迭代过程 1--center
            # new的k/2+i = new k/2+i的左边 + 原来的k/2+i
            # new的k/2-i = new k/2-i的右边 + 原来的k/2-i
            # 偏移来自上一个核元素的约束
            # 行 
            offset[:,:,center,center+index,:,:] = offset[:,:,center,center+index-1,:,:]+ offset[:,:,center,center+index,:,:]
            offset[:,:,center,center-index,:,:] = offset[:,:,center,center-index+1,:,:]+ offset[:,:,center,center-index,:,:]
            # 列
            offset[:,:,center+index,center,:,:] = offset[:,:,center+index-1,center,:,:]+ offset[:,:,center+index,center,:,:]
            offset[:,:,center-index,center,:,:] = offset[:,:,center-index+1,center,:,:]+ offset[:,:,center-index,center,:,:]
       
 
    
        offset = offset.reshape(N,self.group*self.kernel_size*self.kernel_size*2,H,W) # b,g*kw*kh*2,h,w

        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

    
          x = sdconv_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm): # dim=out-dim=64
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H,W
        """
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] 0::2 从0开始每隔2个选一个，1::2 从1开始每隔2个选一个
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] 在x3的维度拼接
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x).view(B, H // 2, W // 2, -1)  # [B, H/2*W/2, 2*C]
        x = x.permute(0, 3, 1, 2)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim // group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ComplexConv2d(nn.Module):
    """ 这是 复数卷积 操作
    output 是特征融合后的复数"""

    def __init__(self, channels, H, W, dtype=torch.complex64):
        super(ComplexConv2d, self).__init__()
        self.dtype = dtype
        # gcc_dk 是指，实现卷积；并融合中间分支的全局信息 
        # 卷积的实部虚部化分开
        self.conv_r1 = gcc_dk(channels, 'W', W, H // 2 + 1)  
        self.conv_i1 = gcc_dk(channels, 'H', W, H // 2 + 1)
        self.conv_i2 = gcc_dk(channels, 'W', W, H // 2 + 1)
        self.conv_r2 = gcc_dk(channels, 'H', W, H // 2 + 1) 
        self.weights_h = nn.Parameter(torch.randn(channels, 1, H // 2 + 1))
        self.weights_w = nn.Parameter(torch.randn(channels, W, 1))
        self.H = H // 2 + 1
        self.W = W

    def forward(self, input):
        a = self.conv_r1(input.real) 
        B, C, _, _ = a.shape
        b = self.conv_r2(input.real).expand(B, C, self.W, self.H)
        a = a.expand(B, C, self.W, self.H)

        a = self.weights_w * a + self.weights_h * b
        b = self.weights_w * self.conv_r1(input.imag).expand(B, C, self.W, self.H) + self.weights_h * self.conv_r2(
            input.imag).expand(B, C, self.W, self.H)
        c = self.weights_h * self.conv_i1(input.real).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.real).expand(B, C, self.W, self.H)
        d = self.weights_h * self.conv_i1(input.imag).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.imag).expand(B, C, self.W, self.H)

        real = (a - c)
        imag = (b + d)
        return real.type(self.dtype) + 1j * imag.type(self.dtype)

class gcc_dk(nn.Module):
    """卷积后，按元素相乘中间分支提取的全局信息，并输出"""
    def __init__(self, channel, direction, W, H):
        super(gcc_dk, self).__init__()
        self.direction = direction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = BAM(channel, W, H) # 中间提特征的分支 c*1*1
       
        self.kernel_generate_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 1), padding=(0, 0), bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), padding=(0, 0), bias=False, groups=channel),
        )
        

    def forward(self, x):
    
        # glob_info为[1,c,1,1]
        glob_info = self.att(x) #中间分支提出的 全局信息
        # H_info[1, c, h, 1]
        H_info = torch.mean(x, dim=1, keepdim=True) #取mean，保留H高
        H_info = self.kernel_generate_conv(H_info) #卷积操作
        #  kernel_input[1, c, h, 1]
        kernel_input = H_info * glob_info.expand_as(H_info) #H_info [1,c,h,1]和global_info扩展成H_info尺寸一致后，按元素相乘，
                                                                # 使用元素相乘融合中间分支的全局信息

        return kernel_input

def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):

    return torch.nn.Conv2d(in_, out, 3, padding=1)




class FD_Moudle(nn.Module):
    """
    输入 空间域x=[b,c,h,w],对x进行完整复数卷积与正则化处理，并输出 空间域x=[b,c,h,w]

    """

    def __init__(self, dim, H, W,stage):
        super().__init__()
        self.f_conv = ComplexConv2d(dim, H, W) #复数卷积
        self.f_bn = CPL.ComplexBatchNorm2d(dim) #复数域的BN 归一化
        self.f_relu = complex_gelu #复数域的gelu
        self.conv = nn.Conv2d(dim, dim, 1)
        group = [8,16,32,32,16,8]
        dia=[3,2,1,1,1,1]      
        self.cross_conv = SDConv_stage(channels=dim,dilation=dia[stage],pad=dia[stage],group=group[stage])# 测试扩张卷积 dilation和pad 保持一致形状就进出一致ok；分阶段控制这个扩张率，
        self.dwconv = nn.Conv2d(in_channels=dim*2,out_channels=dim,kernel_size=1)
 



    def forward(self, x):

        bias = x
        dtype = x.dtype

        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0,2,1).reshape(B,C,H,W)
    
        # shortcut
        x2 = self.conv(x) #bchw
       
#  AF-Conv
        F_x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho") #傅里叶变换

        GF_x = self.f_conv(F_x) 
        complex_GF = self.f_relu(self.f_bn(GF_x)) #融合后的复数特征，作复值bn、复数relu 正则操作

      
        real = (F_x.real*complex_GF.real - F_x.imag*complex_GF.imag) #频域乘法
        imag = (F_x.real * complex_GF.imag + complex_GF.real*F_x.imag)


        complex_GF = real.type(GF_x.dtype) + 1j * imag.type(GF_x.dtype)


       
        F_x = torch.fft.irfft2(complex_GF, s=(H, W), dim=(2, 3), norm="ortho") #具有全局 特征
# ------
        x = x.permute(0,2,3,1) #bhwc
        x = self.cross_conv(x)
        x = x.permute(0,3,1,2) #bchw
        x = torch.cat((x,F_x),dim=1) #channel 拼接 2dim
        x = self.dwconv(x) # 2dim--dim
  


        x = x.reshape(B, C, N).permute(0, 2, 1)
        x = x.type(dtype)
        x2 = x2.reshape(B, C, N).permute(0, 2, 1)
        output = F.gelu(x + x2)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        # hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3 = nn.GroupNorm(out_features // 4, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.gn3(x)
        x = self.drop(x)
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        return x


class FDConv_Module(nn.Module):

    def __init__(self, dim, dim_out, mlp_ratio=0.25, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5,stage=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = FD_Moudle(dim, H=h, W=w,stage=stage)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim_out, act_layer=act_layer,
                       drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim_out)), requires_grad=True) #可学习参数
        self.dim = dim
        self.dim_out = dim_out

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.dim == self.dim_out:
            x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        else:
            x = self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x


class FDConv_Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    # def __init__(self, in_channels, out_channels,stage, mid_channels=None, resolution_H=448, resolution_W=448): #TODO 448改512
    def __init__(self, in_channels, out_channels, stage,mid_channels=None, resolution_H=SIZE, resolution_W=SIZE):

        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            FDConv_Module(in_channels, mid_channels, h=resolution_H, w=resolution_W,stage=stage),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
          
        )

    def forward(self, x):
        return self.double_conv(x)





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out





class decoder(nn.Module):
    def __init__(self, in_channels, out_channels,H,W,stage):
        super(decoder, self).__init__()
        # ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.FDBlock =FDConv_Block(in_channels, out_channels, resolution_H=H, resolution_W=W,stage=stage)
       

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # Iterative filling, in order to obtain better results
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # Different filling volume
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # Splicing
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.FDBlock(out)
        return out_conv

class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W,stage):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Bottleneck1(in_channels, out_channels),
            FDConv_Block(out_channels, out_channels, resolution_H=H, resolution_W=W,stage=stage)
        )

    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, H, W, stage,bilinear=True, ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = FDConv_Block(in_channels, out_channels, resolution_H=H, resolution_W=W,stage=stage)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = PatchExpand(in_channels)
            self.conv = FDConv_Block(in_channels, out_channels, resolution_H=H, resolution_W=W,stage=stage)

    def forward(self, x2, x1):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Bottleneck1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block1, layers, num_classes, bilinear=False):
        self.H = self.W = SIZE 
        self.inplanes = 128
        self.bilinear = bilinear
        super(ResNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        
        )

    
        factor = 2 if bilinear else 1
       

        self.down1 = Down(64, 128, self.H // 2, self.W // 2,stage=0)
        self.down2 = Down(128, 256, self.H // 2, self.W // 2,stage=0)
        self.layer1 = ResDown(256,512,self.W //8, self.H //8,stage=2)
        self.layer2 = ResDown(512, 1024 // factor, self.W // 16, self.H // 16,stage=2)
        self.up0 = decoder(1024, 512,self.H//8, self.W//8,stage = 2)
        self.up1 = decoder(512, 256,self.H//4, self.W//4,stage = 3)

        self.up2 = Up(256, 128 // factor, self.H // 2, self.W // 2,stage=4, bilinear=bilinear)
        self.up3 = Up(128, 64, self.H, self.W, stage=5,bilinear=bilinear)
   
        #Full connection
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initalize_weights()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
 
        x0 = self.conv0(x) #x0 [H W 64]

        x1 = self.down1(x0)# x1 [H/2,W/2,128] 2c

        x2 = self.down2(x1)#  256 4c

        x3 = self.layer1(x2)# x3 [H/8,W/8,512] layer1

        x4 = self.layer2(x3)# x4 [H/16,W/16,1024] layer2


        x = self.up0(x3, x4)
        x = self.up1(x2, x)
        x = self.up2(x1, x)
        x = self.up3(x0, x)

        final = self.final_conv(x)

        return final


from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
@BACKBONES.register_module()
def FDNet(num_classes=2):
    # model = ResNet(Bottleneck, Bottleneck1, [3, 4, 6,3 ], num_classes)
    #test
    model = ResNet(Bottleneck, Bottleneck1, [1, 1, 1,1 ], num_classes)

    return model


