import os
import glob
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Config
# -----------------------
@dataclass
class TrainConfig:
    data_root: str = "Processed"          # carpeta que contiene mayusculas/minusculas/numeros
    subset: str = "minusculas"         # "mayusculas" | "minusculas" | "numeros"
    img_size: int = 32
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 15
    seed: int = 7
    out_model: str = "Models/ocr_cnn_minusculas.pt"
    out_labels: str = "Models/labels_minusculas.txt"


cfg = TrainConfig()


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_gray(path: str):
    # Devuelve None si no puede leer (archivo corrupto/nombre raro/etc)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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
        th = th[y0:y1 + 1, x0:x1 + 1]

    # Pad a cuadrado y resize
    h, w = th.shape
    s = max(h, w)
    pad_y = (s - h) // 2
    pad_x = (s - w) // 2
    th = cv2.copyMakeBorder(
        th,
        pad_y, s - h - pad_y,
        pad_x, s - w - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    th = cv2.resize(th, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # [0,1]
    th = th.astype(np.float32) / 255.0
    return th


# -----------------------
# Dataset
# -----------------------
def list_valid_classes(root_dir: str):
    """
    Filtra carpetas/clases válidas:
    - mayusculas: A..Z
    - minusculas: a..z
    - numeros: 0..9
    """
    norm = root_dir.replace("\\", "/").lower()

    classes = []
    for d in os.listdir(root_dir):
        full = os.path.join(root_dir, d)
        if not os.path.isdir(full):
            continue

        # MAYÚSCULAS
        if norm.endswith("/mayusculas"):
            if len(d) == 1 and d.isalpha() and d.upper() == d:
                classes.append(d)

        # MINÚSCULAS
        elif norm.endswith("/minusculas"):
            if len(d) == 1 and d.isalpha() and d.lower() == d:
                classes.append(d)

        # NÚMEROS
        elif norm.endswith("/numeros"):
            if len(d) == 1 and d.isdigit():
                classes.append(d)

        # Por si usas otra carpeta
        else:
            classes.append(d)

    return sorted(classes)


class CharFolderDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int, labels=None, augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        # ✅ AQUÍ está el cambio importante:
        classes = list_valid_classes(root_dir)

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
                self.samples.extend(
                    [(p, self.class_to_idx[c]) for p in glob.glob(os.path.join(root_dir, c, f"*.{ext}"))]
                )

        if not self.samples:
            raise RuntimeError(f"No hay imágenes en {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Reintenta si hay imágenes corruptas/ilegibles
        for _ in range(10):
            path, y = self.samples[idx]
            img = read_gray(path)

            if img is None:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            # Augmentación ligera
            if self.augment:
                if random.random() < 0.35:
                    ang = random.uniform(-10, 10)
                    h, w = img.shape
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

                if random.random() < 0.25:
                    img = cv2.GaussianBlur(img, (3, 3), 0)

            x = preprocess_char(img, self.img_size)
            x = torch.tensor(x).unsqueeze(0)  # (1,H,W)
            return x, torch.tensor(y, dtype=torch.long)

        raise RuntimeError("Demasiadas imágenes corruptas/ilegibles en el dataset.")


# -----------------------
# Model
# -----------------------
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


# -----------------------
# Train
# -----------------------
def main():
    set_seed(cfg.seed)

    # ✅ crea "Models" (la tuya) si no existe
    os.makedirs("Models", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = os.path.join(cfg.data_root, cfg.subset)
    all_classes = list_valid_classes(root)

    print("Dataset:", root)
    print("Clases detectadas:", all_classes)
    print("Total clases:", len(all_classes))

    ds_full = CharFolderDataset(root, cfg.img_size, labels=all_classes, augment=True)

    # Split train/val
    random.shuffle(ds_full.samples)
    n = len(ds_full.samples)
    n_val = int(0.1 * n)
    val_samples = ds_full.samples[:n_val]
    train_samples = ds_full.samples[n_val:]

    train_ds = ds_full
    train_ds.samples = train_samples

    val_ds = CharFolderDataset(root, cfg.img_size, labels=all_classes, augment=False)
    val_ds.samples = val_samples

    # ✅ en Windows mejor 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

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
