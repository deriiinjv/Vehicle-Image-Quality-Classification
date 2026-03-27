import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn
import os
import pandas as pd

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model
model = efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 3)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

#tta
tta_transforms = [
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])
]

#opencv
def get_opencv_scores(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_norm = min(1.0, blur_score / 200)
    brightness = gray.mean()
    light_norm = min(1.0, brightness / 100)
    edge_score = np.sum(cv2.Canny(gray, 100, 200) > 0)
    edge_norm = min(1.0, edge_score / 50000)
    return {
        "blur": 1 - blur_norm,
        "low_light": 1 - light_norm,
        "good": (blur_norm + light_norm + edge_norm) / 3
    }

#dl tta
def get_dl_probs_tta(img):
    probs_list = []
    for t in tta_transforms:
        img_t = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        probs_list.append(probs)
    mean_probs = np.mean(probs_list, axis=0)
    return {
        "blur": float(mean_probs[0]),
        "good": float(mean_probs[1]),
        "low_light": float(mean_probs[2])
    }

#fusion
def combine_scores(dl, cv):
    return {k: 0.7 * dl[k] + 0.3 * cv[k] for k in dl}

#threshold
THRESHOLD = 0.8

#decision
def final_decision(scores):
    label = max(scores, key=scores.get)
    confidence = scores[label]
    if label == "good" and confidence > THRESHOLD:
        return label, confidence, "Accept"
    return label, confidence, "Reject"

#process
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    dl = get_dl_probs_tta(img)
    cv = get_opencv_scores(img)
    final = combine_scores(dl, cv)
    return final_decision(final)

#run
def run_inference(input_dir, output_file, threshold):
    global THRESHOLD
    THRESHOLD = threshold
    results = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            img_path = os.path.join(root, file)
            res = process_image(img_path)
            if res is None:
                continue
            label, confidence, status = res
            results.append({
                "image_filename": file,
                "predicted_label": label,
                "confidence_score": float(confidence),
                "status": status
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


#main
if __name__ == "__main__":
    input_dir = "processed_data/val"
    output_file = "results.csv"
    threshold = 0.8

    run_inference(input_dir, output_file, threshold)