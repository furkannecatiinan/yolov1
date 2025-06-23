import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')  # use sum, not mean
        self.S = S  # grid size (7x7)
        self.B = B  # boxes per cell
        self.C = C  # number of classes
        self.lambda_coord = lambda_coord  # weight for box loss
        self.lambda_noobj = lambda_noobj  # weight for no-object confidence loss

    def forward(self, pred, target):
        # pred & target shape: (batch_size, S, S, 30)

        # mask: where there is an object (confidence > 0)
        obj_mask = target[..., 4] > 0
        noobj_mask = target[..., 4] == 0

        #1. Localization loss (x, y, w, h), only for object cells
        pred_boxes = pred[obj_mask][..., :5]       # x, y, w, h, conf
        target_boxes = target[obj_mask][..., :5]

        # (x, y) position loss
        xy_loss = self.mse(pred_boxes[..., 0:2], target_boxes[..., 0:2])

        # (w, h) size loss, use sqrt to reduce effect of large boxes
        wh_loss = self.mse(torch.sqrt(torch.abs(pred_boxes[..., 2:4] + 1e-6)),
                           torch.sqrt(torch.abs(target_boxes[..., 2:4] + 1e-6)))

        loc_loss = xy_loss + wh_loss

        # 2. Confidence loss
        # confidence loss for object cells
        conf_obj = self.mse(pred[obj_mask][..., 4], target[obj_mask][..., 4])
        # confidence loss for empty cells
        conf_noobj = self.mse(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])

        # 3. Classification loss (only if object exists)
        class_loss = self.mse(pred[obj_mask][..., 5:], target[obj_mask][..., 5:])

        # Total loss
        total = (self.lambda_coord * loc_loss +        # box location
                 conf_obj +                            # confidence (object)
                 self.lambda_noobj * conf_noobj +      # confidence (no object)
                 class_loss)                           # class probabilities

        return total
