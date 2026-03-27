from inference import process_image
import os

# sample folder
folder = "processed_data/val"

print("Testing images...\n")

for cls in os.listdir(folder):
    class_path = os.path.join(folder, cls)

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)

        result = process_image(img_path)

        print(f"{file} → {result}")