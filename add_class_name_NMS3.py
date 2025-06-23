"""
confidence x class probabilities = box score
elimination of same class boxes with NMS (Non-Maximum Suppression)
add name of the class to top of the box
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

S, B, C = 7, 2, 20
image_size = 448
cell_size = image_size // S
conf_threshold = 0.1
iou_threshold = 0.4

boxes = []

class_names = [f'Class {i}' for i in range(C)] # Dummy class names

# First, add the boxes

yolo_output = np.zeros((S, S, B * 5 + C))  # 5 for bounding box (x, y, w, h, confidence) and C for class probabilities
yolo_output[3, 4, 0:5] = [0.5, 0.5, 0.3, 0.4, 0.9]  # Box1
yolo_output[3, 4, 10:30] = np.eye(C)[2]  # Class 2 = 100% (one-hot)

image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

for i in range(S):
    for j in range(S):
        class_probs = yolo_output[i, j, B * 5:]
        class_id = np.argmax(class_probs)
        class_score = class_probs[class_id]

        for b in range(B):
            base = b * 5
            conf = yolo_output[i, j, base + 4]
            score = conf * class_score

            if score > conf_threshold:
                x = yolo_output[i, j, base + 0]
                y = yolo_output[i, j, base + 1]
                w = yolo_output[i, j, base + 2]
                h = yolo_output[i, j, base + 3]

                abs_x = (j + x) * cell_size
                abs_y = (i + y) * cell_size
                abs_w = w * image_size
                abs_h = h * image_size

                x1 = int(abs_x - abs_w / 2)
                y1 = int(abs_y - abs_h / 2)
                x2 = int(abs_x + abs_w / 2)
                y2 = int(abs_y + abs_h / 2)

                boxes.append([x1, y1, x2, y2, score, class_id])

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

final_boxes = []
boxes.sort(key=lambda x: x[4], reverse=True)  # Sort by score

while boxes:
    best = boxes.pop(0)
    final_boxes.append(best)
    boxes = [box for box in boxes if box[5] != best[5] or iou(box, best) < iou_threshold]

for x1, y1, x2, y2, score, class_id in final_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_names[class_id]}: {score:.2f}"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('YOLO with NMS & Labels')
plt.axis('off')
plt.show()