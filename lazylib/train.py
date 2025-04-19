import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, roc_curve, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from .now import now
from tqdm import tqdm
from .device import device
from torch import nn

def train(epoch:int, optimizer, loss_fn, MODEL:nn.Module, dataset:DataLoader) -> dict:
    # Put the MODEL in the training mode    
    MODEL.train()
    y_true = []
    y_scores = []
    running_loss = 0.0
    correct = 0
    total = 0
    ym = ([], [])
    qt = 0

    it = tqdm(enumerate(dataset), total=len(dataset))

    for _, (x, y) in it:
        x = x.to(device())
        y = y.to(device())
                
        # Make predictions for this batch
        output = MODEL(x)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        y_true.extend(y.cpu().numpy())
        y_scores.extend(output[:, 1].cpu().detach().numpy())  # Assuming output[:, 1] is probability of positive class
        # Adjust learning weights
        optimizer.step()
        correct += torch.sum(torch.argmax(output, 1).eq(y)).item()
        total += y.size(0)
        qt += len(x)
        ym[0].extend(torch.argmax(output, 1).cpu().numpy())
        ym[1].extend(y.data.cpu().numpy())

        # Gather data and report
        running_loss += loss.item()
        n = now()
        it.set_description(f"[{n}] Epoch {str(epoch).zfill(3)} Acc: {correct/qt:.4f} Loss: {running_loss / len(dataset):.8f}")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # Loss / Accuracy
    return dict({
            "epoch": epoch,
            "vsal_loss": running_loss / len(dataset),
            "val_accuracy": correct / total,
            "roc_curve": (fpr, tpr, thresholds),
            "auc": auc,
            "metricsFromY":ym,
            "confusion_matrix":confusion_matrix(y_true, ym[0])
        })