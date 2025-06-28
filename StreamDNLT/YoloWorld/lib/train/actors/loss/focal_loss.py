import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入和目标都应为 [N, C] 的形状
        # inputs 是未经过 sigmoid 的 logits
        # targets 是标签，形状为 [N]
        
        # 计算概率
        probs = torch.sigmoid(inputs)
        
        # 计算交叉熵
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # 计算 Focal Loss
        alpha_t = self.alpha
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        




class FocalLossCost(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction=None):
        super(FocalLossCost, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入和目标都应为 [N, C] 的形状
        # inputs 是未经过 sigmoid 的 logits
        # targets 是标签，形状为 [N]
        
        
        
        # 计算概率
        probs = torch.sigmoid(inputs)
        
        # 计算交叉熵
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # 计算 Focal Loss
        alpha_t = self.alpha
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss