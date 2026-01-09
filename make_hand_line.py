import os, random, cv2, numpy as np

TEXT = "HOLA"
ROOT = r"Processed\mayusculas"   # cambia a Processed\minusculas o Processed\numeros si quieres
OUT = "test_hand.png"
SPACE = 12

random.seed(7)

imgs = []
for ch in TEXT:
    folder = os.path.join(ROOT, ch)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not files:
        raise FileNotFoundError(f"No hay imágenes en {folder}")
    f = random.choice(files)
    im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
    imgs.append(im)

h = max(im.shape[0] for im in imgs)
w = sum(im.shape[1] for im in imgs) + SPACE*(len(imgs)-1)
canvas = np.ones((h, w), dtype=np.uint8) * 255

x = 0
for im in imgs:
    canvas[0:im.shape[0], x:x+im.shape[1]] = im
    x += im.shape[1] + SPACE

cv2.imwrite(OUT, canvas)
print("Generada:", OUT)
