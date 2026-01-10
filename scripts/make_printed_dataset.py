import os
import random
import cv2
import numpy as np

# Clases a generar (puedes ampliar)
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")

# Fuentes internas de OpenCV (sin instalar fuentes externas)
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SIMPLEX,
]

def render_char(ch, img_size=64):
    # lienzo blanco grande
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255

    font = random.choice(FONTS)
    scale = random.uniform(1.2, 2.4)
    thickness = random.randint(2, 5)

    # calcula tamaño del texto para centrar
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    x = (img_size - tw) // 2 + random.randint(-3, 3)
    y = (img_size + th) // 2 + random.randint(-3, 3)

    # dibuja en negro
    cv2.putText(canvas, ch, (x, y), font, scale, (0,), thickness, cv2.LINE_AA)

    # pequeñas variaciones (ruido / blur / rotación)
    if random.random() < 0.5:
        ang = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((img_size/2, img_size/2), ang, 1.0)
        canvas = cv2.warpAffine(canvas, M, (img_size, img_size), borderValue=255)

    if random.random() < 0.3:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    if random.random() < 0.3:
        noise = np.random.normal(0, 8, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return canvas

def main():
    out_root = os.path.join("Printed", "chars")  # dataset impreso
    os.makedirs(out_root, exist_ok=True)

    # crea carpetas por clase
    for c in CLASSES:
        os.makedirs(os.path.join(out_root, c), exist_ok=True)

    n_per_class = 1200  # total ~ (36 * 1200) = 43k imágenes

    for c in CLASSES:
        for i in range(n_per_class):
            img = render_char(c, img_size=64)

            # Normaliza a 32x32 estilo tu pipeline (binariza/crop/pad/resize)
            gray = cv2.GaussianBlur(img, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            ys, xs = np.where(bw > 0)
            if len(xs) and len(ys):
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                crop = bw[y0:y1+1, x0:x1+1]
            else:
                crop = bw

            h, w = crop.shape
            s = max(h, w)
            top = (s - h) // 2
            bottom = s - h - top
            left = (s - w) // 2
            right = s - w - left
            square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            square = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

            # vuelve a negro sobre blanco
            out_img = 255 - square

            path = os.path.join(out_root, c, f"{c}_{i:05d}.png")
            cv2.imwrite(path, out_img)

    print("✅ Dataset impreso creado en:", out_root)

if __name__ == "__main__":
    main()
