import cv2
import numpy as np

img = np.ones((220, 1200, 3), dtype=np.uint8) * 255
cv2.putText(img, "ABCD XYZ 012345", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,0), 4)
cv2.imwrite("test_line.png", img)
print("Generada: test_line.png")
