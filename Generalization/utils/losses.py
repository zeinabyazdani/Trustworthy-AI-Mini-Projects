import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        n_classes = logits.size(1)
        pred = logits.log_softmax(dim=1)
        smooth_labels = torch.zeros_like(pred)
        smooth_labels.fill_(self.label_smoothing / n_classes)
        smooth_labels.scatter_(1, target.data.unsqueeze(1), (1 - self.label_smoothing))
        loss = torch.sum(-smooth_labels * pred, dim=1).mean()
        return loss
    

if __name__ == '__main__':
    # Test
    logits = torch.randn(1, 10)
    target = torch.randint(10, size=[1])
    # logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    # targets = torch.tensor([0, 1])
    
    criterion1 = LabelSmoothingCrossEntropy(label_smoothing=0.1)
    criterion2 = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(criterion1(logits, target), criterion2(logits, target))
