import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn

# ---- mismo modelo que train.py ----
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

def preprocess_page(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarización robusta
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)

    # Une trazos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, th

def extract_components(th):
    # Encuentra contornos = posibles caracteres
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    H, W = th.shape
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # filtros (ajusta según tu dataset)
        if area < 60:
            continue
        if h < 8 or w < 3:
            continue
        if h > 0.9 * H:
            continue

        boxes.append((x, y, w, h))

    # Orden aproximado por líneas (y) y por x
    # Agrupación simple: ordena por y, luego x
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def crop_and_normalize(gray, box, img_size):
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]

    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # tinta blanca
    if np.mean(th) < 127:
        th = 255 - th

    ys, xs = np.where(th > 0)
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        th = th[y0:y1+1, x0:x1+1]

    h2, w2 = th.shape
    s = max(h2, w2)
    pad_y = (s - h2) // 2
    pad_x = (s - w2) // 2
    th = cv2.copyMakeBorder(th, pad_y, s - h2 - pad_y, pad_x, s - w2 - pad_x,
                            borderType=cv2.BORDER_CONSTANT, value=0)
    th = cv2.resize(th, (img_size, img_size), interpolation=cv2.INTER_AREA)
    th = (th.astype(np.float32) / 255.0)
    x = torch.tensor(th).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return x

def group_lines(boxes, y_thresh=18):
    # agrupa por líneas según centro y
    lines = []
    for b in boxes:
        x, y, w, h = b
        cy = y + h / 2
        placed = False
        for line in lines:
            if abs(line["cy"] - cy) < y_thresh:
                line["boxes"].append(b)
                # actualiza cy promedio
                line["cy"] = (line["cy"] * (len(line["boxes"]) - 1) + cy) / len(line["boxes"])
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "boxes": [b]})

    # ordena líneas y ordena cajas por x
    lines.sort(key=lambda d: d["cy"])
    for line in lines:
        line["boxes"].sort(key=lambda b: b[0])
    return lines

def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    labels = ckpt["labels"]
    img_size = ckpt["img_size"]
    model = SimpleCNN(n_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, labels, img_size

def ocr_image(image_path, model_path, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    model, labels, img_size = load_model(model_path)

    gray, th = preprocess_page(img)
    boxes = extract_components(th)
    lines = group_lines(boxes)

    results = []
    text_lines = []
    with torch.no_grad():
        for li, line in enumerate(lines):
            line_text = ""
            for b in line["boxes"]:
                x_tensor = crop_and_normalize(gray, b, img_size)
                logits = model(x_tensor)
                pred = int(torch.argmax(logits, dim=1).item())
                ch = labels[pred]
                line_text += ch

                results.append({
                    "char": ch,
                    "box": {"x": int(b[0]), "y": int(b[1]), "w": int(b[2]), "h": int(b[3])},
                    "line": li
                })
            text_lines.append(line_text)

    base = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(out_dir, base + ".txt")
    json_path = os.path.join(out_dir, base + ".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image": image_path, "lines": text_lines, "items": results}, f, ensure_ascii=False, indent=2)

    return txt_path, json_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True, help="models/ocr_cnn.pt")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    txt, js = ocr_image(args.image, args.model, args.out)
    print("✅ OCR guardado en:")
    print(" -", txt)
    print(" -", js)
