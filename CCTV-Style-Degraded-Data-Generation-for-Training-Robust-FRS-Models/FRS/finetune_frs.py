import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1

# --- 1. CONFIGURATION AND HYPERPARAMETERS ---
DATA_DIR = 'D:/DBDA_Project/Datasets/HQ_Train_Dataset'
NUM_CLASSES = 971  # Adjust this to the number of classes in your dataset
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Number of epochs for training the new layers
LEARNING_RATE = 0.01

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')


# --- 2. DATA LOADING AND PREPROCESSING ---
def get_dataloaders(data_dir, batch_size):
    """
    Sets up the data transformations and creates data loaders for training and validation.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    return train_dataloader, val_dataloader, dataset_sizes


# --- 3. MODEL DEFINITION ---
class FineTunedModel(nn.Module):
    """
    A custom model that combines a pre-trained InceptionResnetV1 with a new
    custom classification head.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Load the pre-trained InceptionResnetV1 model
        base_model_full = InceptionResnetV1(pretrained='vggface2')

        # We manually collect the layers we want to keep from the base model
        # The forward pass of the InceptionResnetV1 model is hardcoded, so we must
        # create a new module with the layers up to the final convolutional block.
        layers_to_keep = []
        found_final_pool = False
        for name, child in base_model_full.named_children():
            if name == 'avgpool_1a':
                found_final_pool = True

            if not found_final_pool:
                layers_to_keep.append(child)

        self.base_model = nn.Sequential(*layers_to_keep)

        # Define the custom classification layers
        
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


# --- 4. TRAINING AND VALIDATION LOGIC ---
def train_and_validate(model, train_loader, val_loader, dataset_sizes, criterion, optimizer, num_epochs, scheduler):
    """
    Runs the training and validation loops for the model.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataloaders = {'train': train_loader, 'val': val_loader}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                # scheduler.step()  # Adjust learning rate
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]) * 100.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if phase == 'train':
            scheduler.step()            # Moved here after optimizer.step()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # 1. Load data
    train_loader, val_loader, dataset_sizes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    class_names = train_loader.dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")

    # 2. Instantiate model
    model = FineTunedModel(NUM_CLASSES).to(device)

    # --- Training Phase: Only train the new layers ---
    print("\n--- Training custom layers ---")

    # Selectively unfreeze the final blocks of the base model for fine-tuning
    for name, param in model.base_model.named_parameters():
        if 'Block8' in name or 'mixed_7a' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Only train the parameters of the new custom layers
    for param in model.added_layers.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = train_and_validate(model, train_loader, val_loader, dataset_sizes, criterion, optimizer, NUM_EPOCHS, scheduler)

    # --- 6. Save the final model ---
    torch.save(model.state_dict(), 'inceptionResNetV1.pth')
    print("Fine Tuned model saved")

