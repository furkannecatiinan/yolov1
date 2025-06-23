# YOLOv1 From Scratch

This project implements YOLOv1 step by step using PyTorch, with minimal external libraries.

## Files Overview

* `input_logic1.py`
  → Creates YOLO output logic (grid, boxes, class scores)

* `draw_box2.py`
  → Visualizes predicted bounding boxes on an image

* `add_class_name_NMS3.py`
  → Applies confidence threshold, Non-Max Suppression, and draws class names

* `yolov1_cnn4.py`
  → Defines the YOLOv1 CNN model

* `loss_function5.py`
  → Implements the YOLOv1 loss function

* `training_loop6.py`
  → Trains the model using dummy data and loss function

* `inferences7.py`
  → Runs inference and shows output on dummy or real images

* `dataset_dataloader8.py`
  → Loads and processes real images and annotations (Pascal VOC format)

## Training Pipeline

1. **Model**: Built with Conv + FC layers to match YOLOv1 output shape `(7, 7, 30)`
2. **Loss**: Custom loss function handles box, confidence, and class loss
3. **Training**: Dummy loop provided; can be replaced with real data
4. **Inference**: Outputs boxes and labels on sample images
5. **Visualization**: Uses OpenCV + Matplotlib to draw results
6. **Dataset**: Converts Pascal VOC `.xml` annotations into YOLO format

## How to Run

1. Train with `training_loop6.py`
2. Test with `inferences7.py`
3. Visualize with `add_class_name_NMS3.py`
4. Use real dataset via `dataset_dataloader8.py`

