import numpy as np
import matplotlib.pyplot as plt


"""
this module includes the logic for yolo's input processing.
7*7 grid,
448*448 px input image

no model is used here, just simulating the output tensor
the output tensor shape is (7, 7, 30) where 30 = 2*5 + 20
2 is the number of bounding boxes per grid cell,
5 is the number of bounding box attributes (x, y, w, h, confidence),
20 is the number of classes.
the output tensor is generated randomly to simulate the model's output.
"""

# dummy image

image_size = 448
dummy_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

S, B, C = 7, 2, 20  # S: grid size, B: number of bounding boxes per grid cell, C: number of classes
output_tensor = np.zeros((S, S, B * 5 + C))  # 5 for bounding box (x, y, w, h, confidence) and C for class probabilities

for i in range(S):
    for j in range(S):
        for b in range(B):
            # Assign random bounding box coordinates and confidence
            output_tensor[i, j, b * 5: b*5+5] = np.random.rand(5)

        class_probs = np.random.rand(C)
        output_tensor[i, j, B * 5:] = class_probs / np.sum(class_probs)  # Normalize class probabilities

print("Output tensor shape:", output_tensor.shape)