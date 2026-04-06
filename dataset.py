import os
from glob import glob
from typing import Optional, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class NIHCardioDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Build a map from filename -> full path (recursive search)
        pattern = os.path.join(images_root, "**", "*.png")
        all_imgs = glob(pattern, recursive=True)
        self.path_map: Dict[str, str] = {os.path.basename(p): p for p in all_imgs}

        missing = 0
        for name in self.df["Image Index"].astype(str).tolist():
            if name not in self.path_map:
                missing += 1

        if missing > 0:
            print(f"[WARN] Missing {missing} images referenced in CSV (out of {len(self.df)})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_name = str(row["Image Index"])
        y = torch.tensor(row["target"], dtype=torch.float32)

        img_path = self.path_map.get(img_name, None)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}. Check images_root.")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, y, img_name