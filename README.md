# 📷 Camera Forensics AI

A robust PyTorch and Streamlit-based system to identify the camera model used to capture an image. Instead of relying on easily forged EXIF metadata, this project analyzes intrinsic signal-level artifacts—specifically the **Photo Response Non-Uniformity (PRNU)**—which acts as a unique sensor noise fingerprint left by the camera hardware.

## 🚀 Features
* **Metadata-Independent Forensics:** Identifies camera models strictly using microscopic pixel noise.
* **Open-Set Anomaly Detection:** Detects and flags images from unknown camera sensors.
* **Deep Feature Extraction:** Uses a customized ResNet-18 architecture built in PyTorch to classify geometric noise syntax.
* **Wavelet-based Scene Removal:** Aggressively isolates the silicon manufacturing dust (noise) by filtering out physical scenery.
* **Sleek Web Interface:** Provides an interactive Streamlit UI for uploading images, viewing original captures vs. extracted noise fingerprints, and observing real-time AI inferences.

## 🧠 Pipeline Overview
1. **Scene Removal (Wavelet Filter):** The system applies a Daubechies Wavelet filter to destroy the physical scenery from the image.
2. **PRNU Extraction:** By subtracting the filtered image from the original, we isolate the invisible Photo Response Non-Uniformity.
3. **Patch Generation:** The pure isolated sensor-noise is sliced into independent 32x32 tensor patches.
4. **Deep Inference:** The patches are fed through a PyTorch ResNet-18 Neural Network to match the noise patterns. A Majority Voting algorithm makes the final prediction.

## 📁 Repository Structure
```
camera_forensics/
│
├── dataset/                     # Split dataset (train/val/test) - Auto-generated
├── dataset_raw/                 # Place raw class folders here for training
├── models/                      # Saved PyTorch models and class mapping (.pth, .txt)
├── src/
│   ├── data_split.py            # Splits dataset_raw into train/val/test splits
│   ├── model.py                 # Defines the ResNet-18 architecture
│   ├── patch_generation.py      # Slices residuals into 32x32 patches
│   ├── predict.py               # Inference script using majority voting
│   ├── preprocessing.py         # Image ingestion, resizing, and center cropping
│   ├── residual_extraction.py   # Daubechies Wavelet transform for noise isolation
│   └── train.py                 # Main training loop script
│
├── app.py                       # Streamlit web application frontend
└── requirements.txt             # Project dependencies
```

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/) (Ensure your version matches your CUDA environment if utilizing GPU)

### 2. Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

*(Requirements include `torch`, `torchvision`, `streamlit`, `scikit-image`, `opencv-python-headless`, `PyWavelets`, `pillow` and others.)*

## 🏋️ Training the Model

### 1. Data Preparation
Place your raw image dataset inside the `dataset_raw/` directory. Each camera model should have its own sub-folder.
```text
dataset_raw/
├── iPhone_13/
│   ├── img1.jpg
...
└── Galaxy_S22/
    ├── img1.jpg
...
```

### 2. Split the Dataset
Split the dataset into training, validation, and testing sets:
```bash
python src/data_split.py
```
This script will populate the `dataset/` directory.

### 3. Start Training
To extract noise prints, generate cropped patches, and train the Deep Neural Network:
```bash
python src/train.py
```
The trained weights (`forensics_cnn.pth`) and class identifiers (`class_mapping.txt`) will be saved in the `models/` directory natively.

## 🌐 Running the Streamlit App

To launch the graphical web interface and interactively test images against your trained model:

```bash
streamlit run app.py
```
Open your browser to the URL provided in your terminal (usually `http://localhost:8501`).

## 🛡️ Important Forensic Notes
* **Lossless Handling:** The Streamlit app converts and saves inputted images losslessly as `.png` files under the hood. Compressing files dynamically as JPEG during testing alters the fragile microscopic noise patterns and heavily corrupts prediction accuracy.

## 🛠️ Built With
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Streamlit](https://streamlit.io/) - Web Frontend Application Framework
- [PyWavelets](https://pywavelets.readthedocs.io/) - Wavelet Transforms (Daubechies)
- [OpenCV](https://opencv.org/) - Mathematical image transformations
