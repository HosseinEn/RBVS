import torch
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryDiceCoefficient

def jaccard_index(SR, GT, threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()
    
    intersection = torch.sum(SR * GT)
    union = torch.sum(SR + GT) - intersection
    jaccard = intersection / (union + 1e-6)
    
    return jaccard.item()

def get_metrics(SR, GT, threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    accuracy = BinaryAccuracy(device='cuda')
    precision = BinaryPrecision(device='cuda')
    recall = BinaryRecall(device='cuda')
    f1_score = BinaryF1Score(device='cuda')
    dice = BinaryDiceCoefficient(device='cuda')

    # Update metrics with predictions and targets
    accuracy.update(SR, GT)
    precision.update(SR, GT)
    recall.update(SR, GT)
    f1_score.update(SR, GT)
    dice.update(SR, GT)

    # Compute the metric values
    acc = accuracy.compute().item()
    se = recall.compute().item()  # Sensitivity is Recall
    sp = accuracy.compute().item()  # Specificity
    pc = precision.compute().item()
    f1 = f1_score.compute().item()
    dc = dice.compute().item()
    js = jaccard_index(SR, GT)  # Jaccard Index

    return acc, se, sp, pc, f1, js, dc