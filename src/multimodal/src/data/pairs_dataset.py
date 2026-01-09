import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TCGAPairsDataset(Dataset):
    """Paired dataset:
    - img: patch image file (png/jpg) OR a directory containing patches (randomly picks one file)
    - omics: precomputed fixed-dim vector saved as .npy
    """

    def __init__(self, split_csv: str, data_root: str, img_size: int = 224):
        self.df = pd.read_csv(split_csv)
        required = {"case_id", "label", "img_path", "omics_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"split_csv missing columns: {sorted(missing)}; required={sorted(required)}")

        self.data_root = data_root
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def _resolve(self, p: str) -> str:
        # allow absolute paths in csv; otherwise join with data_root
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def _pick_patch(self, img_path: str) -> str:
        p = self._resolve(img_path)
        if os.path.isdir(p):
            # choose any image inside
            candidates = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                candidates.extend(list(sorted(__import__("glob").glob(os.path.join(p, ext)))))
            if not candidates:
                raise RuntimeError(f"No image patches found in dir: {p}")
            # deterministic-ish: use hash to pick stable patch per epoch if desired; here random
            return np.random.choice(candidates)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(row["label"])

        img_file = self._pick_patch(row["img_path"])
        img = Image.open(img_file).convert("RGB")
        x_img = self.tf(img)

        omics_file = self._resolve(row["omics_path"])
        vec = np.load(omics_file).astype("float32")
        x_omics = torch.from_numpy(vec)

        return x_img, x_omics, label
