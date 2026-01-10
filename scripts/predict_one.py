import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

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

def preprocess_char(img_gray: np.ndarray, img_size: int) -> np.ndarray:
    img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = 255 - th

    ys, xs = np.where(th > 0)
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        th = th[y0:y1+1, x0:x1+1]

    h, w = th.shape
    s = max(h, w)
    pad_y = (s - h) // 2
    pad_x = (s - w) // 2
    th = cv2.copyMakeBorder(th, pad_y, s - h - pad_y, pad_x, s - w - pad_x,
                            borderType=cv2.BORDER_CONSTANT, value=0)
    th = cv2.resize(th, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return th.astype(np.float32) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    print("Imagen:", args.image)
    print("Modelo:", args.model)

    ckpt = torch.load(args.model, map_location="cpu")
    labels = ckpt["labels"]
    img_size = ckpt["img_size"]

    model = SimpleCNN(n_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    print("img is None?", img is None)
    if img is None:
        raise FileNotFoundError(f"No pude leer la imagen: {args.image}")

    x = preprocess_char(img, img_size)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    print("Predicción:", labels[pred])

if __name__ == "__main__":
    main()
