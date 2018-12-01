"""
BBox transform
"""

import torch


def bbox_transform(boxes, gtboxes):
    """ Bounding Box Transform
    from groundtruth boxes and proposal boxes to deltas

    Args:
        boxes: [N, 4] torch.Tensor (xyxy)
        gtboxes: [N, 4] torch.Tensor  (xywh)
    Return:
        delta: [N, 4] torch.Tensor
    """
    gt_w = gtboxes[:, 2] - gtboxes[:, 0] + 1
    gt_h = gtboxes[:, 3] - gtboxes[:, 1] + 1
    # center
    gt_x = gtboxes[:, 0] + 0.5 * gt_w
    gt_y = gtboxes[:, 1] + 0.5 * gt_h

    # Anchors [x,y,w,h]
    anchor_x = boxes[:, 0]
    anchor_y = boxes[:, 1]
    anchor_w = boxes[:, 2]
    anchor_h = boxes[:, 3]
    # anchor_w = boxes[:, 2] - boxes[:, 0] + 1
    # anchor_h = boxes[:, 3] - boxes[:, 1] + 1
    # # center
    # anchor_x = boxes[:, 0] + 0.5 * anchor_w
    # anchor_y = boxes[:, 1] + 0.5 * anchor_h

    delta_x = (gt_x - anchor_x) / anchor_w
    delta_y = (gt_y - anchor_y) / anchor_h
    delta_w = torch.log(gt_w / anchor_w)
    delta_h = torch.log(gt_h / anchor_h)

    # [N, 4]
    return torch.stack([delta_x, delta_y, delta_w, delta_h]).transpose(0, 1)


def bbox_transform_inv(boxes, delta):
    """ Inverse Bounding Box Transform
    from deltas and proposal boxes to predicted boxes
    Args:
        boxes: [N, 4] torch.Tensor (xywh)
        delta: [N, 4] torch.Tensor (xywh)
    Return:
        pred: [N, 4] torch.Tensor (xyxy)
    """
    pred_boxes = torch.zeros_like(boxes)
    pred_x = boxes[:, 0] + boxes[:, 2] * delta[:, 0]
    pred_y = boxes[:, 1] + boxes[:, 3] * delta[:, 1]
    pred_w = boxes[:, 2] * torch.exp(delta[:, 2])
    pred_h = boxes[:, 3] * torch.exp(delta[:, 3])

    pred_boxes[:, 0] = pred_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_y + 0.5 * pred_h

    return pred_boxes


if __name__ == '__main__':

    pass
