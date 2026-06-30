"""
Loss functions & Metrics cho binary segmentation.
"""
import torch
import torch.nn as nn


# ===== Metrics =====

def dice_coefficient(pred, target, threshold=0.5, epsilon=1e-6):
    """Tính Dice coefficient (F1 cho segmentation)."""
    pred = (pred > threshold).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)


def compute_metrics(outputs, masks, threshold=0.5, epsilon=1e-6):
    """Tính IoU, Accuracy, Recall, Precision."""
    preds = (outputs > threshold).float().view(-1)
    masks = masks.view(-1)

    TP = (preds * masks).sum()
    TN = ((1 - preds) * (1 - masks)).sum()
    FP = (preds * (1 - masks)).sum()
    FN = ((1 - preds) * masks).sum()

    iou = (TP + epsilon) / (TP + FP + FN + epsilon)
    accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)
    recall = (TP + epsilon) / (TP + FN + epsilon)
    precision = (TP + epsilon) / (TP + FP + epsilon)

    return iou.item(), accuracy.item(), recall.item(), precision.item()


# ===== Loss Functions =====

class DiceLoss(nn.Module):
    """Dice Loss = 1 - Dice coefficient."""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Kết hợp BCE Loss + Dice Loss."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)
