import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from finetune_frs import FineTunedModel  # Replace with actual script/module if separate

# --- CONFIGURATION ---
TEST_DATA_DIR = 'D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/FRS Model/'  # Set to your degraded test dataset path
MODEL_PATH = 'inceptionResNetV1.pth'
BATCH_SIZE = 32
NUM_CLASSES = 971  # Same as training

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --- DATA TRANSFORM ---
test_transforms = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- LOAD DATA ---
test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- LOAD MODEL ---
model = FineTunedModel(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- EVALUATE ON TEST DATA ---
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'\nTest Accuracy on Degraded Dataset: {test_accuracy:.2f}%')
