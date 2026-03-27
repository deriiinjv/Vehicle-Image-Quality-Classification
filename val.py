import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
import torch.nn as nn
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
if not os.path.exists("processed_data/val"):
    print("Dataset not found. Skipping validation.")
    exit()
val_data = datasets.ImageFolder("processed_data/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=32)

model = efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 3)

model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)