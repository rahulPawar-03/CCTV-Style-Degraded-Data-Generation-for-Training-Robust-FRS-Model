# CCTV-Style Degraded Data Generation for Training Robust FRS Models

## 📌 Overview
This project provides a **data degradation pipeline** designed to simulate **CCTV-style low-quality images** for training **robust Face Recognition Systems (FRS)**.  
By applying a variety of controlled degradations to high-quality face images, the model learns to identify individuals even under poor acquisition conditions.

The pipeline processes a dataset of face images and outputs degraded variants in a structured format suitable for training deep learning models.

---

## 🎯 Objectives
- Generate realistic **low-quality CCTV-style images** from clean face datasets.
- Support **multiple degradation categories** (sensor noise, compression artifacts, transmission losses, etc.).
- Enable **custom severity control** for each degradation type.
- Maintain **train/validation directory structure** for direct model training.

---

## 🛠 Features
- Group A: Acquisition & Sensor Defects: (Incorrect White Balance, Under Exposure, Over Exposure, Vignetting, Chromatic Aberration) - These degradations simulate issues that arise at the point of capture due to camera hardware limitations or settings.

- Group B: Transmission & Compression Artifacts: (Bitrate Starvation, Blockiness, Packet Loss, Color Banding) - This group focuses on visual artifacts introduced during the encoding, compression, and transmission of video data.

- Group C: Blurring Effects: (Motion Blur, Gaussian Blur, Lens Blur) - These degradations simulate the loss of image sharpness and detail due to motion or optical limitations.

- Group D: Noise Injection: (Gaussian Noise, Salt & Pepper Noise) - This group models random visual noise, often a result of low light conditions or sensor quality.


---

## 📂 Project Structure
```
MAIN_FOLDER/
│
├── train/
│   ├── Person_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── Person_2/
│       ├── img1.jpg
│       └── img2.jpg
│
└── val/
    ├── Person_1/
    │   ├── img1.jpg
    └── Person_2/
        ├── img1.jpg
```
Degraded outputs are saved in a **parallel folder** structure.

---

## 📦 Installation
```bash
# Clone this repository
git clone https://github.com/rahulPawar-03/CCTV-Style-Degraded-Data-Generation-for-Training-Robust-FRS-Model.git
cd CCTV-Style-Degraded-Data-Generation

```

---

## 🚀 Usage
```bash
python degrade_images.py     --input_dir path/to/train     --output_dir path/to/output_train     --severity mild
```

**Parameters:**
- `--input_dir` : Path to input dataset folder.
- `--output_dir` : Path to save degraded images.
- `--severity` : One of `mild`, `moderate`, `strong`.

---

## ⚙️ Example Weighted Severity Values
| Effect                  | Value  |
|-------------------------|--------|
| White Balance Strength  | 1.3    |
| Under Exposure Alpha    | 0.5    |
| Over Exposure Alpha     | 1.6    |
| Vignetting Strength     | 0.5    |
| Chromatic Aberration    | 6      |
| Blockiness Applied      | Yes    |
| JPEG Quality            | 10     |
| Packet Loss Probability | 0.03   |
| Color Banding Levels    | 8      |
| Gaussian Blur           | Applied|

---

## 🧠 Model Training Use Case
The degraded images can be used alongside clean images to train models like:
- **InceptionResnetV1** (FaceNet)
- **ArcFace**
- **MobileFaceNet**

This improves real-world robustness for surveillance and low-quality camera scenarios.

---

## 🤝 Contributing
Contributions are welcome!  
You can:
- Suggest new degradation types.
- Improve parameter tuning.
- Add GPU acceleration.

---

