import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex, Dice

# SR : Segmentation Result
# GT : Ground Truth

def get_metrics(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    accuracy = Accuracy().to('cuda')
    precision = Precision().to('cuda')
    recall = Recall().to('cuda')
    f1_score = F1Score().to('cuda')
    jaccard = JaccardIndex(num_classes=2).to('cuda')
    dice = Dice().to('cuda')

    acc = accuracy(SR.int(), GT.int())
    se = recall(SR.int(), GT.int())  # Sensitivity is Recall
    sp = accuracy(SR.int() == 0, GT.int() == 0)  # Specificity
    pc = precision(SR.int(), GT.int())
    f1 = f1_score(SR.int(), GT.int())
    js = jaccard(SR.int(), GT.int())
    dc = dice(SR.int(), GT.int())

    return acc, se, sp, pc, f1, js, dc