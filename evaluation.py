# evaluation.import torch
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex, BinaryDiceCoefficient

def get_metrics(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    accuracy = BinaryAccuracy().to('cuda')
    precision = BinaryPrecision().to('cuda')
    recall = BinaryRecall().to('cuda')
    f1_score = BinaryF1Score().to('cuda')
    jaccard = BinaryJaccardIndex().to('cuda')
    dice = BinaryDiceCoefficient().to('cuda')

    acc = accuracy(SR, GT)
    se = recall(SR, GT)  # Sensitivity is Recall
    sp = accuracy(SR == 0, GT == 0)  # Specificity
    pc = precision(SR, GT)
    f1 = f1_score(SR, GT)
    js = jaccard(SR, GT)
    dc = dice(SR, GT)

    return acc.item(), se.item(), sp.item(), pc.item(), f1.item(), js.item(), dc.item()
