# Weapon Detection using YOLOv8

This project demonstrates how to fine-tune a YOLOv8 model for the task of weapon detection in images. The goal is to create a system that can accurately identify and locate various types of weapons, which could be a critical component in security and surveillance systems.

## 📜 Project Overview

The project leverages the state-of-the-art YOLOv8 object detection model and fine-tunes it on a custom dataset of weapon images. The entire pipeline, from data acquisition to model training and inference, is implemented in a single Jupyter Notebook.

### Key Features:

* **YOLOv8 Model**: Utilizes the powerful and efficient YOLOv8s (small) model as a base for transfer learning.
* **Roboflow Dataset**: The training data is sourced from a curated dataset on Roboflow, which simplifies the process of data acquisition and annotation.
* **End-to-End Pipeline**: The notebook covers all necessary steps, including setting up the environment, downloading data, configuring the model, training, and running predictions on a test set.

## ⚙️ Methodology

1.  **Environment Setup**:
    * The necessary libraries, `ultralytics`, `roboflow`, and `pyyaml`, are installed.
    * The Roboflow API is used to download the "wepons-detection-p5lkk" dataset.

2.  **Data Configuration**:
    * A `data.yaml` file is created to configure the dataset paths, number of classes, and class names for the YOLOv8 training process. The classes include "gun", "knife", "pistol", "rifle", etc.

3.  **Model Training**:
    * The YOLOv8s model, pre-trained on the COCO dataset, is loaded.
    * The model is then trained on the custom weapon dataset for 30 epochs with an image size of 640x640. Data augmentation is enabled during training to improve model robustness.

4.  **Inference and Evaluation**:
    * After training, the best-performing model weights are loaded.
    * The model is used to run predictions on the test images provided in the dataset.
    * The results, including bounding boxes drawn on the images, are saved for review.

## 📊 Results

The model was trained for 30 epochs, and the validation metrics show promising performance in detecting various weapon classes. The final mean Average Precision (mAP50-95) on the validation set was **0.285**.

Key metrics for some of the classes include:
* **person**: mAP50-95 of 0.706
* **rifle**: mAP50-95 of 0.697
* **weapon**: mAP50-95 of 0.376
* **Knife**: mAP50-95 of 0.344

## 🚀 How to Use

1.  **API Key**: You will need a Roboflow API key to download the dataset. Replace the placeholder in the code with your own key.
2.  **Run Notebook**: Execute the single cell in the `Yolov8WeponsDetection.ipynb` notebook. It will automatically handle the entire process from setup to prediction.
3.  **View Results**: The prediction results will be saved in the `runs/detect/predict` directory.