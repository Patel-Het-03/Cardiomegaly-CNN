import os
import json
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd

from dataset import NIHCardioDataset

@dataclass
class Config:
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    images_root: str = "data/images"
    out_dir: str = "outputs/logs"
    model_out: str = "models/best_model.pt"
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 5
    img_size: int = 224
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = (2 * precision * recall) / max((precision + recall), 1e-9)
    return {"acc": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_y = []
    all_p = []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits)
        all_y.append(y.cpu())
        all_p.append(probs.cpu())
    y_true = torch.cat(all_y).numpy().astype(int)
    y_prob = torch.cat(all_p).numpy()
    metrics = compute_metrics(y_true, y_prob)
    return metrics, y_true, y_prob

def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = NIHCardioDataset(cfg.train_csv, cfg.images_root, transform=train_tf)
    val_ds = NIHCardioDataset(cfg.val_csv, cfg.images_root, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model: ResNet18 pretrained, replace final layer to 1 logit
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(cfg.device)

    # Loss (handle imbalance with pos_weight)
    train_df = pd.read_csv(cfg.train_csv)
    pos = train_df["target"].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_f1 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        total_loss = 0.0

        for x, y, _ in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = total_loss / max(len(train_loader.dataset), 1)

        val_metrics, _, _ = run_eval(model, val_loader, cfg.device)

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, cfg.model_out)
            print(f"Saved best model to {cfg.model_out} (val_f1={best_f1:.4f})")

    # Save training history
    hist_path = os.path.join(cfg.out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print("Saved history:", hist_path)

if __name__ == "__main__":
    main()