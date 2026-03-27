import cv2
import matplotlib.pyplot as plt
from inference import process_image

img_path = "processed_data/val/good/-_-_2007_14-_H180_V0_JPG.rf.ea1c6c9e25d058a0f8a0b90257c80c10.jpg.jpg"

img = cv2.imread(img_path)

label, conf, status = process_image(img_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

color = "green" if status == "Accept" else "red"

plt.imshow(img_rgb)
plt.title(f"{label} | {conf:.2f} | {status}", color=color)
plt.axis("off")
plt.show()
