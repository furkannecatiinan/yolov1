"""
YOLOv1 CNN Model 
This code implements a simple CNN model for YOLOv1.
It includes the model architecture, input processing, and a dummy output tensor to simulate the model's output.
The model is designed to work with a 448x448 input image and a 7x7 grid.
The output tensor shape is (7, 7, 30) where 30 = 2*5 + 20.
2 is the number of bounding boxes per grid cell,
5 is the number of bounding box attributes (x, y, w, h, confidence),
20 is the number of classes.
The output tensor is generated randomly to simulate the model's output.

No training is performed in this code.
It is a simple demonstration of how to set up the model and process the input.
The model can be extended to include training and evaluation logic as needed.
"""

import torch 
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S  # number of grid cells (S x S)
        self.B = B  # number of boxes per grid cell
        self.C = C  # number of classes

        # Convolutional feature extractor (based on YOLOv1 paper)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # downsample: 448 -> 224
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            nn.Conv2d(192, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Usually repeated blocks here in original paper
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),

            nn.AdaptiveAvgPool2d((7, 7)),  # final pooling to get 7x7 feature maps
        )

        # Fully connected layers (predict bounding boxes and class scores)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten all feature maps
            nn.Linear(1024 * S * S, 4096),  # first FC layer
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # helps prevent overfitting
            nn.Linear(4096, S * S * (B * 5 + C))  # final output (7x7x30)
        )

    def forward(self, x):
        x = self.features(x)           # extract spatial features
        x = self.classifier(x)         # convert to prediction vector
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)  # reshape to (batch, 7, 7, 30)
        return x

# Example usage
if __name__ == "__main__":
    model = YOLOv1()
    dummy_input = torch.randn(1, 3, 448, 448)  # batch=1, RGB image 448x448
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expected: (1, 7, 7, 30)
# Output shape: (1, 7, 7, 30)