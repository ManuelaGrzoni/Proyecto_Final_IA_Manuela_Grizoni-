import os
import glob
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
@dataclass
class TrainConfig:
    data_root = "Images"      
    subset = "mayusculas"          
    img_size: int = 32
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 12
    seed: int = 7
    out_model = "Models/ocr_cnn_mayusculas.pt"
    out_labels = "Models/labels_mayusculas.txt"

cfg = TrainConfig()

# Utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def preprocess_char(img_gray: np.ndarray, img_size: int) -> np.ndarray:
    # Normaliza a "tinta blanca" sobre fondo negro (mejor para CNN)
    img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Si el fondo sale negro, invertimos
    if np.mean(th) < 127:
        th = 255 - th

    # Recorta bbox del contenido
    ys, xs = np.where(th > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        th = th[y0:y1+1, x0:x1+1]

    # Pad a cuadrado y resize
    h, w = th.shape
    s = max(h, w)
    pad_y = (s - h) // 2
    pad_x = (s - w) // 2
    th = cv2.copyMakeBorder(th, pad_y, s - h - pad_y, pad_x, s - w - pad_x,
                            borderType=cv2.BORDER_CONSTANT, value=0)
    th = cv2.resize(th, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # [0,1]
    th = (th.astype(np.float32) / 255.0)
    return th

# Dataset
class CharFolderDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int, labels=None, augment=False):
        self.img_size = img_size
        self.augment = augment

        # Clases: subcarpetas
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if labels is None:
            self.labels = classes
        else:
            self.labels = labels

        self.class_to_idx = {c: i for i, c in enumerate(self.labels)}

        self.samples = []
        for c in classes:
            if c not in self.class_to_idx:
                continue
            for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                for p in glob.glob(os.path.join(root_dir, c, f"*.{ext}")):
                    self.samples.append((p, self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No hay imágenes en {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = read_gray(path)

        # Augmentación ligera (sin “magia” OCR)
        if self.augment:
            if random.random() < 0.35:
                ang = random.uniform(-10, 10)
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderValue=255)

            if random.random() < 0.25:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        x = preprocess_char(img, self.img_size)  # (H,W) float
        x = torch.tensor(x).unsqueeze(0)         # (1,H,W)
        return x, torch.tensor(y, dtype=torch.long)


# Model
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    set_seed(cfg.seed)
    os.makedirs("models", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.subset == "mixed":
        roots = [os.path.join(cfg.data_root, "printed"), os.path.join(cfg.data_root, "handwritten")]
    else:
        roots = [os.path.join(cfg.data_root, cfg.subset)]

    # Construye lista de etiquetas global (por orden)
    all_classes = None
    for r in roots:
        classes = sorted([d for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))])
        all_classes = classes if all_classes is None else sorted(list(set(all_classes).union(classes)))

    # Dataset combinado simple (concat samples)
    datasets = [CharFolderDataset(r, cfg.img_size, labels=all_classes, augment=True) for r in roots]
    # Merge manual
    samples = []
    for ds in datasets:
        samples.extend(ds.samples)
    merged = datasets[0]
    merged.samples = samples
    merged.labels = all_classes
    merged.class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Split train/val
    random.shuffle(merged.samples)
    n = len(merged.samples)
    n_val = int(0.1 * n)
    val_samples = merged.samples[:n_val]
    train_samples = merged.samples[n_val:]

    train_ds = merged
    train_ds.samples = train_samples
    val_ds = CharFolderDataset(roots[0], cfg.img_size, labels=all_classes, augment=False)
    val_ds.samples = val_samples

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(n_classes=len(all_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, total)

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / max(1, total)

        print(f"Epoch {ep:02d} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "labels": all_classes, "img_size": cfg.img_size}, cfg.out_model)

    with open(cfg.out_labels, "w", encoding="utf-8") as f:
        for c in all_classes:
            f.write(c + "\n")

    print(f"\n✅ Mejor val_acc={best_acc:.4f}")
    print(f"Modelo guardado en: {cfg.out_model}")
    print(f"Etiquetas en: {cfg.out_labels}")

if __name__ == "__main__":
    main()
