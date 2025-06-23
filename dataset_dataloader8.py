import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #  1. Load image 
        image_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (448, 448))
        image = image[:, :, ::-1]  # BGR to RGB
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        #  2. Load annotation 
        label_filename = image_filename.replace(".jpg", ".xml")
        label_path = os.path.join(self.label_dir, label_filename)
        boxes = self._parse_voc(label_path)

        #  3. Convert to YOLO output tensor 
        y = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box in boxes:
            class_id, x, y_center, w, h = box
            i = int(y_center * self.S)
            j = int(x * self.S)
            x_cell = x * self.S - j
            y_cell = y_center * self.S - i
            w_cell = w
            h_cell = h

            y[i, j, 0:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell, 1])
            y[i, j, 10 + class_id] = 1  # one-hot class

        return image, y

    def _parse_voc(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall("object"):
            class_id = self._class_to_id(obj.find("name").text)
            bbox = obj.find("bndbox")
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)

            x_center = ((x1 + x2) / 2) / 448
            y_center = ((y1 + y2) / 2) / 448
            w = (x2 - x1) / 448
            h = (y2 - y1) / 448
            boxes.append([class_id, x_center, y_center, w, h])

        return boxes

    def _class_to_id(self, class_name):
        class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                      'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']
        return class_list.index(class_name)

"""
That is the example of a dataset dataloader for YOLO input processing.
It reads images and their corresponding VOC XML annotations, processes them into a format suitable for YOLO
training, and returns the images and their corresponding YOLO output tensors.
Requires dataset structure:
dataset/
    images/
        image1.jpg
        image2.jpg
        ...
    labels/
        image1.xml
        image2.xml
This code assumes the VOC dataset format, where each image has a corresponding XML file with bounding box annotations.
The bounding boxes are normalized to the range [0, 1] relative to the image size of 448x448 pixels.
The output tensor shape is (7, 7, 30) where 30 = 2*5 + 20
(2 bounding boxes per grid cell, 5 attributes per box, and 20 classes).
The class names are mapped to indices based on the VOC dataset class list.
The dataset can be used with PyTorch's DataLoader for training YOLO models.     
"""