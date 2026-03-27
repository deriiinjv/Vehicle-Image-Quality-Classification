import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn
from utils import get_opencv_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL
model = efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 3)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# TTA
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

def combine_scores(dl, cv):
    return {k: 0.7 * dl[k] + 0.3 * cv[k] for k in dl}

def final_decision(scores):
    label = max(scores, key=scores.get)
    confidence = scores[label]

    if label == "good" and confidence > 0.8:
        return label, confidence, "Accept"
    return label, confidence, "Reject"

def process_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    dl = get_dl_probs_tta(img)
    cv = get_opencv_scores(img)

    final = combine_scores(dl, cv)

    return final_decision(final)