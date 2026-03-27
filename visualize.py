import cv2
import matplotlib.pyplot as plt
from inference import process_image

img_path = "processed_data/val/good/sample.jpg"

img = cv2.imread(img_path)

label, conf, status = process_image(img_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

color = "green" if status == "Accept" else "red"

plt.imshow(img_rgb)
plt.title(f"{label} | {conf:.2f} | {status}", color=color)
plt.axis("off")
plt.show()