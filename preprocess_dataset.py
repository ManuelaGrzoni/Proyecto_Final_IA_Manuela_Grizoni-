import os
import shutil
import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def normalize_to_32x32(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # tinta -> blanco (255), fondo -> negro (0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        # nada detectado
        out = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return out

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = bw[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

    # vuelve a negro sobre blanco (más “humano” y consistente)
    out = 255 - out
    return out

def main():
    src_root = os.path.join("Images", "numeros")
    dst_root = os.path.join("Processed", "numeros")
    os.makedirs(dst_root, exist_ok=True)

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    classes = [c for c in classes if len(c) == 1 and c.isdigit()]
    classes.sort()

    total_ok = 0
    total_fail = 0

    for c in classes:
        in_dir = os.path.join(src_root, c)
        out_dir = os.path.join(dst_root, c)
        os.makedirs(out_dir, exist_ok=True)

        for fn in os.listdir(in_dir):
            if not fn.lower().endswith(IMG_EXTS):
                continue

            src_path = os.path.join(in_dir, fn)
            norm = normalize_to_32x32(src_path)
            if norm is None:
                total_fail += 1
                continue

            out_name = os.path.splitext(fn)[0] + ".png"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, norm)
            total_ok += 1

    print("Procesadas OK:", total_ok)
    print("Fallaron lectura:", total_fail)
    print("Salida:", dst_root)

if __name__ == "__main__":
    main()
