import os
import cv2
import numpy as np
import argparse
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def preprocess_page(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    return gray, th

def extract_components(th):
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = th.shape
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 60: 
            continue
        if h < 8 or w < 3:
            continue
        if h > 0.9 * H:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])  # asume una sola línea
    return boxes

def normalize_char(gray, box, img_size=32):
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    out = 255 - th  # negro sobre blanco
    return out

def class_folder_for_char(ch):
    if ch.isalpha():
        if ch.isupper():
            return "mayusculas", ch
        else:
            return "minusculas", ch
    if ch.isdigit():
        return "numeros", ch
    raise ValueError(f"Carácter no soportado: {ch}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="ruta a la imagen (BuenasNoches.png)")
    ap.add_argument("--text", required=True, help="texto exacto en la imagen (ej. BuenasNoches)")
    ap.add_argument("--out_root", default="Processed", help="carpeta Processed")
    ap.add_argument("--prefix", default="user", help="prefijo para nombres de archivos")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
        raise FileNotFoundError(img_path)

    text = args.text.strip()
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError("No pude leer la imagen")

    gray, th = preprocess_page(img)
    boxes = extract_components(th)

    # --- Guardar visualización con cajas ---
    vis = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, max(0, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    os.makedirs("outputs_demo", exist_ok=True)
    out_boxes = os.path.join("outputs_demo", f"{img_path.stem}_boxes.png")
    cv2.imwrite(out_boxes, vis)
    print("🧾 Boxes guardadas en:", out_boxes)

    # Filtra el texto quitando espacios (si tu imagen no tiene espacios)
    text_no_spaces = text.replace(" ", "")

    if len(boxes) != len(text_no_spaces):
        print("⚠️ No coincide el número de cajas con el número de letras.")
        print("Cajas detectadas:", len(boxes))
        print("Letras en texto:", len(text_no_spaces))
        print("Sugerencia: ajusta filtros o usa --text sin espacios.")
        # seguimos pero guardamos hasta el mínimo
    n = min(len(boxes), len(text_no_spaces))

    saved = 0
    for i in range(n):
        ch = text_no_spaces[i]
        subset, cls = class_folder_for_char(ch)

        out_dir = Path(args.out_root) / subset / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        norm = normalize_char(gray, boxes[i], img_size=32)
        out_name = f"{cls}_{args.prefix}_{img_path.stem}_{i:02d}.png"
        cv2.imwrite(str(out_dir / out_name), norm)
        saved += 1

    print(f"✅ Guardados {saved} caracteres en {args.out_root}/(mayusculas|minusculas|numeros)/...")

if __name__ == "__main__":
    main()
