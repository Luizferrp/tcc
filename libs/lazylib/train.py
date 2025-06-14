import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, roc_curve, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from .now import now
from tqdm import tqdm
from .device import device
from torch import nn

import time
import os
import torch
from torch import nn
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def Epoch(epoch:int, optimizer, loss_fn, MODEL:nn.Module, dataset, device, salvar_em=None, class_names=None):
    pred_labels = []
    true_labels = []
    y_scores = []
    y_true = []
    running_loss = 0.0
    correct = 0
    total = 0

    total_inference_time = 0.0
    total_samples = 0

    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)

    it = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}")

    for _, (x, y) in it:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Medir tempo de inferência
        start_time = time.time()
        output = MODEL(x)
        end_time = time.time()

        total_inference_time += (end_time - start_time)
        total_samples += x.size(0)

        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true.extend(y.cpu().numpy())
        y_scores.extend(probs[:, 1].detach().cpu().numpy())
        pred_labels.extend(pred.cpu().numpy())
        true_labels.extend(y.cpu().numpy())

        correct += (pred == y).sum().item()
        total += y.size(0)
        running_loss += loss.item()

        it.set_postfix(acc=correct/total, loss=running_loss/len(dataset))

    # Métricas
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, pos_label=1)
    prec = precision_score(true_labels, pred_labels, pos_label=1)
    rec = recall_score(true_labels, pred_labels, pos_label=1)
    roc = roc_auc_score(true_labels, y_scores)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
    fpr, tpr, _ = roc_curve(true_labels, y_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(true_labels, y_scores)

    avg_inference_time = total_inference_time / total_samples
    avg_inference_str = f"{avg_inference_time * 1e6:.3f} µs"


    return {
        "epoch": epoch,
        "loss": running_loss / len(dataset),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "classification_report": report,
        "Precision-Recall": (pr_precision, pr_recall, pr_thresholds),
        "inference_total_time": total_inference_time,
        "inference_avg_time_per_sample": avg_inference_time,
        "avg_inference_str": avg_inference_str
    }

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt
import os

def Epoch_anal(run_dict, class_names=None, salvar_em=None):
    """
    Exibe métricas e gráficos de avaliação para uma época de treinamento.
    
    Parâmetros:
    - run_dict: dicionário com métricas e dados retornados por uma época.
    - class_names: nomes das classes (para matriz de confusão).
    - true_labels: rótulos verdadeiros (para curvas ROC e PR).
    - y_scores: probabilidades previstas (para curvas ROC e PR).
    - salvar_em: diretório para salvar os gráficos (opcional).
    """
    # Extrai os dados do dicionário
    epoch = run_dict["epoch"]
    acc = run_dict["accuracy"]
    f1 = run_dict["f1"]
    prec = run_dict["precision"]
    rec = run_dict["recall"]
    roc = run_dict["roc_auc"]
    cm = run_dict["confusion_matrix"]
    report = run_dict["classification_report"]
    
    # Impressão das métricas
    print(f"Epoch {epoch}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(run_dict["avg_inference_str"])
    print(report)

    # Gráfico
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Métricas - Epoch {epoch}", fontsize=16)

    # Matriz de confusão
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[0])
    axes[0].set_title("Matriz de Confusão")

    # Curva ROC
    if run_dict['roc_curve']:
        fpr, tpr = run_dict['roc_curve']
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=axes[1])
        axes[1].set_title("Curva ROC")

        # Precision-Recall
        precision, recall, _= run_dict["Precision-Recall"]
        axes[2].plot(recall, precision)
        axes[2].set_title("Curva Precision-Recall")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
    else:
        axes[1].axis("off")
        axes[2].axis("off")
        axes[1].text(0.5, 0.5, "ROC/PR Curve indisponíveis", ha="center", va="center")
        axes[2].text(0.5, 0.5, "ROC/PR Curve indisponíveis", ha="center", va="center")

    plt.tight_layout()

    # Salva o gráfico se necessário
    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)
        caminho = os.path.join(salvar_em, f"epoch_{epoch}_metricas.png")
        plt.savefig(caminho)
        print(f"Gráfico salvo em: {caminho}")

    plt.show()