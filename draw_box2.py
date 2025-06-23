import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parameters 
S, B, C = 7, 2, 20  # S: grid size, B: number of bounding boxes per grid cell, C: number of classes
image_size = 448  # Size of the input image
cell_size = image_size // S  # Size of each grid cell

# dummy yolo output
yolo_output = np.zeros((S, S, B * 5 + C))  # 5 for bounding box (x, y, w, h, confidence) and C for class probabilities

# blank image
image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

conf_threshold = 0.1  # Confidence threshold for drawing boxes



yolo_output[3, 4, 0:5] = [0.5, 0.5, 0.3, 0.4, 0.9]  # Box1
yolo_output[3, 4, 10:30] = np.eye(C)[2]  # sınıf 2 = %100 (one-hot)



for i in range(S):
    for j in range(S):
        for b in range(B):
            base = b * 5
            conf = yolo_output[i, j, base + 4]

            if conf > conf_threshold:
                # x, y: center of the bounding box relative to the grid cell
                x = yolo_output[i, j, base + 0]
                y = yolo_output[i, j, base + 1]
                w = yolo_output[i, j, base + 2]
                h = yolo_output[i, j, base + 3]

                # Convert to absolute image coordinates
                abs_x = int((j + x) * cell_size)
                abs_y = int((i + y) * cell_size)
                abs_w = int(w * image_size)
                abs_h = int(h * image_size)

                # left, top, right, bottom coordinates
                x1 = int(abs_x - abs_w / 2)
                y1 = int(abs_y - abs_h / 2)
                x2 = int(abs_x + abs_w / 2)
                y2 = int(abs_y + abs_h / 2)

                # draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                





plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('YOLO Bounding Boxes')
plt.axis('off')  # Hide axes
plt.show()
