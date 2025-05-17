"""
Özel kayıp fonksiyonları.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementasyonu.
    Zor örneklere daha fazla odaklanarak sınıflandırma performansını artırır.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CombinedLoss(nn.Module):
    """
    CrossEntropy ve Focal Loss'u birleştiren kayıp fonksiyonu.
    """
    def __init__(self, alpha=None, gamma=2, ce_weight=0.5, reduction='mean'):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction=reduction)
        self.ce_weight = ce_weight
        
    def forward(self, inputs, targets):
        fl = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.ce_weight * ce + (1 - self.ce_weight) * fl 