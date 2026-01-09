import os
import random
import cv2
import numpy as np

def pick_char_image(root, ch):
    folder = os.path.join(root, ch)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No existe la carpeta de clase: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not files:
        raise FileNotFoundError(f"No hay .png en: {folder}")

    f = random.choice(files)
    path = os.path.join(folder, f)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"No pude leer: {path}")
    return img, path

def build_line(text, root, out_path, space=12, pad=12):
    imgs = []
    picked = []

    for ch in text:
        if ch == " ":
            imgs.append(None)
            picked.append((" ", ""))
            continue
        im, src = pick_char_image(root, ch)
        imgs.append(im)
        picked.append((ch, src))

    # altura máxima
    heights = [im.shape[0] for im in imgs if im is not None]
    if not heights:
        raise RuntimeError("Texto vacío o solo espacios.")
    H = max(heights)

    # calcula ancho total
    W = pad * 2
    for im in imgs:
        if im is None:
            W += space * 2  # espacio extra para palabras
        else:
            W += im.shape[1] + space
    W += pad

    canvas = np.ones((H + pad*2, W), dtype=np.uint8) * 255

    x = pad
    y = pad
    for im in imgs:
        if im is None:
            x += space * 2
            continue
        canvas[y:y+im.shape[0], x:x+im.shape[1]] = im
        x += im.shape[1] + space

    cv2.imwrite(out_path, canvas)
    return picked

def main():
    random.seed(7)

    # ---- CONFIGURA AQUÍ TU TEXTO ----
    nombre = "MANUELA"     # pon tu nombre como quieras (solo letras A-Z si usas mayúsculas)
    nums   = "20261234"    # puedes cambiarlo

    # rutas
    root_may = r"Processed\mayusculas"
    root_num = r"Processed\numeros"

    out_name = "demo_nombre.png"
    out_nums = "demo_numeros.png"

    print("Generando:", out_name)
    picked1 = build_line(nombre, root_may, out_name)
    print("OK. Fuentes usadas:")
    for ch, src in picked1:
        if ch != " ":
            print(" ", ch, "->", src)

    print("\nGenerando:", out_nums)
    picked2 = build_line(nums, root_num, out_nums)
    print("OK. Fuentes usadas:")
    for ch, src in picked2:
        if ch != " ":
            print(" ", ch, "->", src)

    print("\nListo:")
    print(" -", out_name)
    print(" -", out_nums)

if __name__ == "__main__":
    main()
