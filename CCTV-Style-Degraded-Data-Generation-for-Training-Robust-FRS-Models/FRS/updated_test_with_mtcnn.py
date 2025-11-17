import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from finetune_frs import FineTunedModel  # Replace with actual script/module if separate

# --- CONFIGURATION ---
TEST_DATA_DIR = 'D:/DBDA_Project/Datasets/train'  # Set to your degraded test dataset path
MODEL_PATH = 'D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/retrain_23_epoch_on_dbca.pth'
BATCH_SIZE = 32
NUM_CLASSES = 971  # Same as training
NUM_WORKERS = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- DATA TRANSFORM ---
test_transforms = transforms.Compose([
    # The ToTensor() transform is removed here because MTCNN already returns a tensor.
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- CUSTOM DATASET CLASS WITH MTCNN ---
class FaceCropDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset, detects faces, and stores cropped images.
        """
        self.data_dir = data_dir
        self.transform = transform
        # The MTCNN is configured to return a pre-processed tensor
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=True, device=device)

        self.image_paths = []
        self.labels = []

        # Load the class names and create a mapping
        self.class_names = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        # Go through all subdirectories (classes) and find image files
        for class_name in self.class_names:
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])

        print(f"Found {len(self.image_paths)} images to process.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Use MTCNN to detect and crop a face
        # The mtcnn object is configured to return a 160x160 cropped face
        cropped_image = self.mtcnn(image)

        # If no face is detected, handle the error gracefully (e.g., return a black image)
        if cropped_image is None:
            # Create a black image tensor as a fallback
            cropped_image_tensor = torch.zeros((3, 160, 160))
            label = self.labels[idx]
            return cropped_image_tensor, label

        # The cropped_image is already a tensor from MTCNN
        # Apply the normalization transform directly to the tensor
        if self.transform is not None:
            cropped_image_tensor = self.transform(cropped_image)
        else:
            cropped_image_tensor = cropped_image

        label = self.labels[idx]

        return cropped_image_tensor, label


if __name__ == '__main__':
    # --- LOAD DATA ---
    test_dataset = FaceCropDataset(TEST_DATA_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
