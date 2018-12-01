"""

Anchor Layer


"""

import torch
import torch.nn.functional as F
from lib.det_ops.anchors import compute_anchor_whs, generate_anchors
import sys
sys.path.append('../')
from lib.bbox import bbox, box_transform
from IPython import embed


class AnchorLayer:
    """ Anchor Layer
    """
    def __init__(self, strides, areas, aspect_ratios, sizes):
        self.aspect_ratios = aspect_ratios
        self.areas = areas
        self.strides = strides
        self.sizes = sizes

        self._anchor_sizes = self._compute_anchor_size()

        # self._num_anchors = len(self.scales) * len(self.aspect_ratios)

    def _compute_anchor_size(self):
        return compute_anchor_whs(len(self.strides), self.areas, self.aspect_ratios, self.sizes)

    def _generate_anchors(self, input_size):
        boxes = generate_anchors(self._anchor_sizes, input_size, self.strides)
        return boxes

    def assign(self, gt_boxes, labels, input_size, neg_thresh=0.4, pos_thresh=0.5):
        """ assign groundtruth box to anchor box

        """
        anchor_boxes = self._generate_anchors(input_size)
        if labels.shape[0] == 0:
            return torch.LongTensor([0]*anchor_boxes.shape[0]), torch.zeros_like(anchor_boxes)
        # M * N
        xyxy_anchors = bbox.xywh2xyxy(anchor_boxes)
        ious = bbox.box_overlaps(xyxy_anchors, gt_boxes)
        max_ious, max_inds = ious.max(1)
        # M * 4
        matched_boxes = gt_boxes[max_inds]
        box_targets = box_transform.bbox_transform(anchor_boxes, matched_boxes)

        cls_targets = labels[max_inds]
        # negative
        cls_targets[max_ious < neg_thresh] = 0
        # ignore
        cls_targets[(max_ious > neg_thresh) & (max_ious < pos_thresh)] = -1
        return cls_targets, box_targets


if __name__ == '__main__':
    # RetinaNet settings
    strides = [8, 16, 32, 64, 128]
    aspect_ratios = [0.5, 1, 2]
    sizes = [1, 2**(1/3), 2**(2/3)]
    areas = [32**2, 64**2, 128**2, 256**2, 512**2]
    anchor_layer = AnchorLayer(strides=strides, areas=areas, aspect_ratios=aspect_ratios, sizes=sizes)

    boxes = torch.Tensor([[10, 20, 44, 56], [50, 34, 260, 340],
                          [70, 80, 190, 410], [360, 270, 500, 600]])
    labels = torch.LongTensor([3, 1, 1, 4])
    cls_target, box_target = anchor_layer.assign(boxes, labels, torch.FloatTensor([600, 600]), 0.4, 0.5)

    embed()
