# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

def hausdorff_distance_3d(tensor1, tensor2):
    # Convert 3D tensors to point sets
    points1 = torch.nonzero(tensor1).to(float)
    points2 = torch.nonzero(tensor2).to(float)
    # print(f'po {points1.shape}')

    distances = []

    for point1 in points1:
        # print(f'type {point1.dtype}')

        min_distance = float('inf')
        for point2 in points2:
            # x = point1 - point2
            # print(f' cha {x.dtype}')
            # d = torch.norm(x)
            distance = torch.norm(point1 - point2)
            min_distance = min(min_distance, distance.item())
        distances.append(min_distance)

    max_distance_1 = max(distances)

    distances = []

    for point2 in points2:
        min_distance = float('inf')
        for point1 in points1:
            distance = torch.norm(point2 - point1)
            min_distance = min(min_distance, distance.item())
        distances.append(min_distance)

    max_distance_2 = max(distances)

    return max(max_distance_1, max_distance_2)


def hausdorff_distance(matrix_a, matrix_b):
    matrix_a = matrix_a.to(float)
    matrix_b = matrix_b.to(float)
    distances_a_to_b = torch.cdist(matrix_a, matrix_b, p=2)  # Euclidean distance
    min_distances_a_to_b, _ = torch.min(distances_a_to_b, dim=1)

    distances_b_to_a = torch.cdist(matrix_b, matrix_a, p=2)  # Euclidean distance
    min_distances_b_to_a, _ = torch.min(distances_b_to_a, dim=1)

    hausdorff_distance = torch.max(torch.max(min_distances_a_to_b), torch.max(min_distances_b_to_a))
    return hausdorff_distance.item()



def soft_erode(img):
    # if len(img.shape)==4:
    #     p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    #     p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    #     return torch.min(p1,p2)
    # elif len(img.shape)==5:
    #     p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
    #     p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
    #     p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
    # return torch.min(torch.min(p1, p2), p3)
    min1=torch.nn.MaxPool2d(
        kernel_size=(3,1),
        stride=1,
        padding=(1,0),
    )
    min2=torch.nn.MaxPool2d(
        kernel_size=(1,3),
        stride=1,
        padding=(0,1),
    )
    return torch.min(min1(-1*img)*-1,min2(-1*img)*-1)
        
    


def soft_dilate(img):
    # if len(img.shape)==4:
    #     return F.max_pool2d(img, (3,3), (1,1), (1,1))
    # elif len(img.shape)==5:
    #     return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))
    max=torch.nn.MaxPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
    )
     
    return   max(img)
def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0



    ####只计算道路1     classes =[0,1]
    for i in [1]:
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],   #[B,H,W]
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            dh = hausdorff_distance(pred[:,i],target[...,i])



            total_loss += dice_loss
            total_loss +=dh # 加入 豪斯多夫距离 作为罚
       
            total_loss += -torch.mean((1 - target[...,i]) * torch.log(1 - pred[:,i] + 1e-6))

    return total_loss 


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    # print(f'pred {pred.shape} target {target.shape} valid {valid_mask.shape}')

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


@weighted_loss
def soft_cldice(y_pred,y_true, smooth=1,iters=3):
        y_pred=y_pred[:, 1]
        y_true=y_true[..., 1]
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true.to(dtype=torch.float32), iters)
        tprec = (torch.sum(torch.mul(skel_pred, y_true))+smooth)/(torch.sum(skel_pred)+smooth)    
        tsens = (torch.sum(torch.mul(skel_true, y_pred))+smooth)/(torch.sum(skel_true)+smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


@LOSSES.register_module()
class CP_ClDiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=0.5,
                 ignore_index=255,
                 loss_name='soft_dice_cldice',
                 **kwargs):
        super(CP_ClDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)###[B,H,W,class]
        valid_mask = (target != self.ignore_index).long()

        loss = (1-self.loss_weight)* dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index) +self.loss_weight*soft_cldice(pred,one_hot_target)
        if loss<0 :
            print(loss)   
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
