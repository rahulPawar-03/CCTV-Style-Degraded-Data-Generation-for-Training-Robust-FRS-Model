import torch
from PIL import Image
from torchvision import transforms, datasets
from FRS.finetune_frs import FineTunedModel  # Relative import

# --- CONFIG ---
TRAIN_DIR = 'D:/DBDA_Project/Datasets/train'
NUM_CLASSES = 971
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- LOAD CLASS NAMES FROM TRAIN FOLDER STRUCTURE ---
class_names = datasets.ImageFolder(TRAIN_DIR).classes


# def load_model(model_choice):
#     if model_choice == "Model A":
#         # Trained on Only High Quality Images
#         # Accuracy on Hq data : 98%
#         # accuracy on A-D Degraded test set : 8%
#         model_path = "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/hq_cropped.pth"
#         train_dir = "D:/DBDA_Project/Datasets/train"
#
#     elif model_choice == "Model B":
#         # Trained on worst Dataset But has low accuracy
#         # Accuracy on degraded trained data : 70%
#         # accuracy on A-D Degraded test set : 32%
#         model_path = "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/epoch_45_005_abcd_dataset.pth"
#         train_dir = "D:/DBDA_Project/Datasets/train"
#
#     else:
#         # Trained on Degraded dataset but as High accuracy
#         # Accuracy on Croped degraded trained data : 83.99%
#         # accuracy on A-D Degraded test set : 39.4%
#         model_path = "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/epoch_23_005_abcd_new_dataset.pth"
#         train_dir = "D:/DBDA_Project/Datasets/train"
#
#     class_names = datasets.ImageFolder(train_dir).classes
#
#     model = FineTunedModel(len(class_names))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     return model.to(device), class_names


# --- MODEL CONFIG MAP ---
MODEL_CONFIGS = {
    "HQ Model": {
        # Trained on Only High Quality Images
        # Accuracy on Hq data : 98%
        # accuracy on A-D Degraded test set : 8%
        "path": "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/hq_cropped.pth",
        "train_dir": "D:/DBDA_Project/Datasets/train"
    },
    "Degraded Dataset Model": {
        # Trained on worst Dataset But has low accuracy
        # Accuracy on degraded trained data : 70%
        # accuracy on A-D Degraded test set : 32%
        "path": "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/epoch_45_005_abcd_dataset.pth",
        "train_dir": "D:/DBDA_Project/Datasets/train"
    },
    "Slightly lesser Degradation Model": {
        # Trained on Degraded dataset but as High accuracy
        # Accuracy on Croped degraded trained data : 83.99%
        # accuracy on A-D Degraded test set : 39.4%
        "path": "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/epoch_23_005_abcd_new_dataset.pth",
        "train_dir": "D:/DBDA_Project/Datasets/train"
    },
    "Retrained Degraded Model" : {
        # Retrained on Reverse order Degradation dataset
        # Accuracy on degraded trained data : 99.94%
        # accuracy on A-D Degraded test set : 55.34%
        "path": "D:/DBDA_Project/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Models/frs_finetune_models/retrain_23_epoch_on_dbca.pth",
        "train_dir": "D:/DBDA_Project/Datasets/train"
    }
}

def load_model(model_choice):
    config = MODEL_CONFIGS[model_choice]
    model_path = config["path"]
    train_dir = config["train_dir"]
    class_names = datasets.ImageFolder(train_dir).classes

    model = FineTunedModel(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device), class_names



def predict_image(pil_img, model_choice):
    image_tensor = transform(pil_img).unsqueeze(0).to(device)
    model, class_names = load_model(model_choice)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence_score = confidence.item()
        # print(f"confidence_score :: {confidence_score}")

    if confidence_score > 0.9 :
        class_idx = predicted.item()
        return class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}", confidence_score*100
    else :
        return  "Unknown", confidence_score


