import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dataset import NIHCardioDataset
from train import run_eval
from metrics_utils import find_best_threshold, compute_auc_roc, compute_confusion

def main():
    ckpt_path = "models/best_model.pt"
    test_csv = "data/splits/test.csv"
    images_root = "data/images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = NIHCardioDataset(test_csv, images_root, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    metrics, y_true, y_prob = run_eval(model, test_loader, device)
    print("Test metrics @0.5 threshold:", metrics)

    # AUC + ROC
    auc, fpr, tpr, roc_thr = compute_auc_roc(y_true, y_prob)
    print("Test ROC-AUC:", float(auc))

    # Best threshold by F1
    best_thr, best_f1, best_prec, best_rec = find_best_threshold(y_true, y_prob)
    print("Best threshold:", float(best_thr))
    print("Best F1:", float(best_f1), "Prec:", float(best_prec), "Rec:", float(best_rec))

    # Confusion at best threshold
    cm = compute_confusion(y_true, y_prob, threshold=best_thr)
    print("Confusion matrix at best threshold:\n", cm)

    # Save predictions
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv("outputs/logs/test_predictions.csv", index=False)

    # Save eval summary for Streamlit
    
    with open("outputs/logs/test_eval_summary.json", "w") as f:
        json.dump(f, indent=2)

    print("Saved outputs/logs/test_eval_summary.json")

if __name__ == "__main__":
    main()