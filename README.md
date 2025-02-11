# Space Debris Detection using YOLOv8

This project aims to develop an automated object detection system to identify space debris using YOLOv8. By leveraging deep learning, the model accurately detects and localizes space debris in satellite images. This solution not only demonstrates the power of modern computer vision techniques but also contributes to safer space operations by enabling early identification of potentially hazardous debris.

---

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Identifying Space Debris in Images](#identifying-space-debris-in-images)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Tools and Technologies](#tools-and-technologies)
- [Challenges Faced](#challenges-faced)
- [Conclusion](#conclusion)
- [How to Run the Project](#how-to-run-the-project)
- [References](#references)

---

## Introduction

Space debris poses a significant risk to operational satellites and manned space missions. With the increasing amount of debris in Earth's orbit, there is a growing need for reliable and automated detection systems. This project utilizes YOLOv8—a state-of-the-art object detection framework—to identify and localize space debris from images captured by space-based sensors, thereby supporting proactive space traffic management.

---

## Objectives

- **Develop a robust detection model:** Leverage YOLOv8 to build a model capable of detecting space debris from satellite images.
- **Real-time detection:** Ensure that the model can process images quickly for potential real-time applications.
- **Accuracy and reliability:** Optimize the model to achieve high accuracy in identifying even small debris objects.
- **Modular and reproducible pipeline:** Create a clear and reproducible workflow from data preprocessing to inference.

---

## Dataset

The dataset is organized into three main folders (already split):
- **train/**: Contains training images and corresponding labels.
- **val/**: Contains validation images and labels for tuning the model.
- **test/**: Contains test images for evaluating the final model performance.

Each image has an associated label file in YOLO format:
```
class_id x_center y_center width height
```
For example, if you have two classes (e.g., `debris_small` and `debris_large`), the YAML configuration file will define:
```yaml
nc: 2
names: ['debris_small', 'debris_large']
```

---

## Identifying Space Debris in Images

The YOLOv8 model processes each image and outputs bounding boxes around detected debris objects. The detection results can be visualized as images with overlaid boxes indicating the location and class of the debris. This visual feedback aids in verifying model performance and understanding detection challenges (such as distinguishing debris from background stars).

---

## Key Features

- **Real-time detection:** Utilizes YOLOv8 for fast and accurate object detection.
- **Standardized dataset structure:** Clear separation of training, validation, and test data.
- **Easy-to-configure pipeline:** Configuration managed via a simple YAML file.
- **Modular code workflow:** Step-by-step implementation from data preprocessing to model inference.
- **Scalability:** Option to upgrade the model (e.g., from yolov8n to yolov8s/m) based on performance and resource requirements.

---

## Methodology

1. **Data Preprocessing:**
   - Resize images to a consistent shape (e.g., 640×640).
   - Normalize pixel values.
   - Ensure label files follow YOLO format.
2. **Model Training:**
   - Configure the dataset paths in `data.yaml`.
   - Train the YOLOv8 model using the Ultralytics package.
   - Monitor training performance using validation metrics.
3. **Model Evaluation:**
   - Evaluate the model on the test set.
   - Measure performance with metrics such as mAP (mean Average Precision).
4. **Inference & Deployment:**
   - Run inference on new images.
   - Optionally export the model for deployment (e.g., ONNX format).

---

## Tools and Technologies

- **Programming Language:** Python
- **Deep Learning Framework:** Ultralytics YOLOv8
- **Image Processing:** OpenCV, Matplotlib, NumPy
- **Development Environment:** Kaggle Notebook (leveraging Kaggle’s dataset integration)

---

## Challenges Faced

- **Small Object Detection:** Detecting small debris objects can be challenging due to limited pixel representation.
- **Dataset Variability:** Variations in lighting, background clutter (e.g., stars, cosmic rays), and debris shape.
- **Computational Resources:** Balancing model complexity and inference speed to ensure real-time performance.

---

## Conclusion

This project demonstrates an effective approach to detecting space debris using modern object detection techniques. The YOLOv8-based model offers a promising solution for enhancing space situational awareness and could serve as a valuable tool in mitigating the risks associated with space debris.

---

## How to Run the Project

1. **Clone the Repository:**
   - Download or clone the project repository to your local machine or Kaggle Notebook.
2. **Setup the Environment:**
   - Install the required libraries:
     ```bash
     pip install ultralytics opencv-python numpy matplotlib
     ```
3. **Configure Dataset Paths:**
   - Update the `data.yaml` file to point to your dataset directory (e.g., `/kaggle/working/datasets` if using Kaggle).
4. **Train the Model:**
   - Run the training script:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolov8n.yaml")
     model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)
     ```
5. **View Results:**
   - Check the output directory (e.g., `runs/detect/train3/exp`) for detection results.
   - Use Matplotlib or IPython’s display functions to visualize the result images inline in your notebook.
6. **Run Inference:**
   - Test the model on new images:
     ```python
     results = model("path/to/test_image.jpg", conf=0.3)
     ```
7. **Optional - Export the Model:**
   - Export to ONNX or another format if deployment is desired:
     ```python
     model.export(format="onnx")
     ```

---

## References

- Ultralytics YOLOv8 Documentation – [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)
- Research articles on space debris detection and object recognition in satellite imagery.
- Kaggle Datasets (if applicable) for space debris imagery.

---
