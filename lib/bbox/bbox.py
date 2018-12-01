"""

Bounding Box


"""

import torch
import numpy as np
from IPython import embed

def np_xywh2xyxy(boxes):
    # [x1,y1,w,h]
    boxes = np.hstack(
        (boxes[:, 0:2], boxes[:, 0:2] + np.maximum(0, boxes[:, 2:4] - 1))
    )

    return boxes


def clip_boxes(boxes, image_height, image_width):
    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=image_width-1)
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=image_width-1)
    boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=image_height-1)
    boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=image_height-1)
    return boxes


def filter_boxes(boxes):

    keep = []
    for i in range(boxes.shape[0]):
        if boxes[i, 0] < boxes[i, 2] and boxes[i, 1] < boxes[i, 3]:
            keep.append(i)
    boxes = boxes[keep]
    return boxes


def xywh2xyxy(boxes):
    """ xywh -> xyxy
    (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height)
    Args:
        boxes: torch.FloatTensor[N,4]
    """

    x1 = boxes[:, 0] - 0.5 * boxes[:, 2]
    y1 = boxes[:, 1] - 0.5 * boxes[:, 3]
    x2 = boxes[:, 0] + 0.5 * boxes[:, 2]
    y2 = boxes[:, 1] + 0.5 * boxes[:, 3]

    return torch.stack([x1, y1, x2, y2]).transpose(0, 1)


def xyxy2xywh(boxes):
    """ xyxy -> xywh
    (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height)
    Args:
        boxes: torch.FloatTensor[N,4]
    """
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    # center
    x = boxes[:, 0] + 0.5 * w
    y = boxes[:, 1] + 0.5 * h

    return torch.stack([x, y, w, h]).transpose(0, 1)


def box_overlaps(box1, box2):
    """ Box IoU(Insertion over Union)(xmin,ymin,xmax,ymax)
    Args:
        box1: torch.FloatTensor[N, 4],
        box2: torch.FloatTensor[M, 4]
        mode: box representation format
    """
    # N = box1.size()[0]
    # M = box2.size()[0]

    # NxMx2
    lo = torch.max(box1[:, None, :2], box2[:, :2])
    hi = torch.min(box1[:, None, 2:], box2[:, 2:])

    inner_rect = (hi - lo + 1).clamp(0)
    # NxMx1
    inner = inner_rect[:, :, 0] * inner_rect[:, :, 1]

    area1 = (box1[:, 2]-box1[:, 0]+1)*(box1[:, 3]-box1[:, 1]+1)
    area2 = (box2[:, 2]-box2[:, 0]+1)*(box2[:, 3]-box2[:, 1]+1)

    iou = inner / (area1[:, None] + area2 - inner)

    return iou


def box_nms(boxes, scores, threshold):
    """Non maximum suppression.
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


if __name__ == '__main__':
    # TODO: Test it!
    pass