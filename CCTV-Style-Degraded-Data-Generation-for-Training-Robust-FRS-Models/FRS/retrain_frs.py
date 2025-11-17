import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1

# --- CONFIGURATION ---
DATA_DIR = 'D:/DBDA_Project/Datasets/HQ_Train_Dataset'  # Path to the new dataset
NUM_CLASSES = 971  # Same as original
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.005

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- DATA LOADING ---
def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    return train_loader, val_loader, sizes

# --- MODEL DEFINITION ---
class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model_full = InceptionResnetV1(pretrained='vggface2')

        layers_to_keep = []
        found_final_pool = False
        for name, child in base_model_full.named_children():
            if name == 'avgpool_1a':
                found_final_pool = True
            if not found_final_pool:
                layers_to_keep.append(child)

        self.base_model = nn.Sequential(*layers_to_keep)

        self.added_layers = nn.Sequential(
            nn.Conv2d(1792, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.added_layers(x)
        return x

# --- TRAINING FUNCTION ---
def train(model, train_loader, val_loader, sizes, criterion, optimizer, scheduler, epochs):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.double() / sizes[phase] * 100
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print()

    print(f'Best val Acc: {best_acc:.2f}%')
    model.load_state_dict(best_model_wts)
    return model

# --- MAIN ---
if __name__ == '__main__':

    pretrained_model_path = "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/epoch_23_005_abcd_new_dataset.pth"
    retrain_model_save_path = "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/'inceptionResNetV1_retrained.pth"

    train_loader, val_loader, sizes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model = FineTunedModel(NUM_CLASSES).to(device)

    # Load previous weights
    model.load_state_dict(torch.load(pretrained_model_path))

    # Train all parameters
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model = train(model, train_loader, val_loader, sizes, criterion, optimizer, scheduler, NUM_EPOCHS)

    torch.save(model.state_dict(), retrain_model_save_path)
    print("Retrained model saved ")
