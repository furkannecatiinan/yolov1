import torch
from torch.utils.data import DataLoader, TensorDataset

from yolov1_cnn4 import YOLOv1       # import YOLOv1 model class
from loss_function5 import YOLOLoss  # import YOLOv1 loss function class


# Create model, loss function, and optimizer
model = YOLOv1()
loss_fn = YOLOLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Generate random dummy data (10 images and targets)
dummy_images = torch.randn(10, 3, 448, 448)        # input images
dummy_targets = torch.randn(10, 7, 7, 30)          # ground truth output (x, y, w, h, conf, classes)

# Wrap in DataLoader for batch processing
dataset = TensorDataset(dummy_images, dummy_targets)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop (1 epoch only)
model.train()  # set model to training mode

for batch_idx, (images, targets) in enumerate(loader):
    preds = model(images)              # forward pass → model prediction
    loss = loss_fn(preds, targets)     # calculate YOLO loss

    optimizer.zero_grad()              # reset gradients
    loss.backward()                    # compute gradients (backprop)
    optimizer.step()                   # update model weights

    print(f"Batch {batch_idx} - Loss: {loss.item():.4f}")  # show current loss
