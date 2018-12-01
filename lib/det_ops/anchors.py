"""
Generate Anchors
"""

import math
import torch


def mesh_grid(x, y):
    """ mesh grid

    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)

    return torch.cat([xx, yy], dim=1).float()


def compute_anchor_whs(num_layers, areas, aspect_ratios, sizes):
    anchors = []
    for i in range(len(areas)):
        area = areas[i]
        for ar in aspect_ratios:
            h = math.sqrt(area / ar)
            w = h * ar
            for s in sizes:
                anchor_h = h * s
                anchor_w = w * s
                anchors.append([anchor_w, anchor_h])
    # M * K * 2
    # Faster R-CNN: 1*K*2 (1x9x2)
    # FPN: 5*K*2 (5x3x2)
    # RetinaNet: 5*K*2 (5*9*2)
    return torch.Tensor(anchors).view(num_layers, -1, 2)


def generate_anchors(anchor_whs, input_size, strides):
    """ generate anchors
    """
    boxes = []
    num_strides = len(strides)
    num_anchors = anchor_whs.shape[1]

    for i in range(num_strides):
        stride = strides[i]
        feature_size = input_size / stride
        fmw, fmh = int(math.ceil(feature_size[0])), int(math.ceil(feature_size[1]))
        xy = mesh_grid(fmh, fmw) + 0.5  # shift to center
        xy = (xy * stride).view(fmh, fmw, 1, 2).expand(fmh, fmw, num_anchors, 2)
        wh = anchor_whs[i].view(1, 1, num_anchors, 2).expand(fmh, fmw, num_anchors, 2)
        box = torch.cat([xy, wh], dim=3)
        boxes.append(box.view(-1, 4))
    boxes = torch.cat(boxes, 0)
    # box: H * W * self._num_anchors * 2
    return boxes
