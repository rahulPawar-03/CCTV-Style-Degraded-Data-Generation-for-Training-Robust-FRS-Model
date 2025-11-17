# model_utils.py
import torch
import torch.nn as nn
import json
from facenet_pytorch import InceptionResnetV1

def load_class_names(path):
    with open(path, 'r') as f:
        return json.load(f)

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
            nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.added_layers(x)
        return x

def load_model_and_classes(model_info):
    model = FineTunedModel(model_info["num_classes"])
    model.load_state_dict(torch.load(model_info["file"], map_location="cpu"))
    model.eval()
    class_names = load_class_names(model_info["class_names"])
    return model, class_names

def preprocess_face(face_image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(face_image)
