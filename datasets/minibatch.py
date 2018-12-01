"""
Create Mini Batch
"""
import cv2
import torch
import random
import numpy as np
from datasets.utils import flip_img_boxes
from lib.det_ops import anchor_target
from datasets.utils import normalize_image, get_im_scale


def create_minibatch_func(config):
    aspect_ratios = config['aspect_ratios']
    anchor_sizes = config['anchor_sizes']
    anchor_areas = config['anchor_areas']
    strides = config['strides']

    anchor_layer = anchor_target.AnchorLayer(aspect_ratios=aspect_ratios,
                                             sizes=anchor_sizes,
                                             areas=anchor_areas,
                                             strides=strides)

    def collate_minibatch(batch):
        # (img, labels, boxes)
        # img: [H, W, C]
        # labels: [N]
        # boxes: [N, 4]
        batch_size = len(batch)
        max_size = config['max_image_size']
        # [N, 1]
        target_size_inds = np.random.randint(
            0, high=len(config['image_scales']), size=batch_size
        )

        image_shapes = np.zeros((batch_size, 2), dtype=np.int)
        image_scales = np.zeros(batch_size, dtype=np.float)
        batch_height = 0
        batch_width = 0
        for i in range(batch_size):
            h, w = batch[i][0].shape[:2]
            target_size = config['image_scales'][target_size_inds[i]]
            h_, w_, s_ = get_im_scale(h, w, target_size, max_size)
            image_shapes[i, 0] = h_
            image_shapes[i, 1] = w_
            image_scales[i] = s_
            batch_height = max(h_, batch_height)
            batch_width = max(w_, batch_width)

        # pad images to support the last stride
        max_stride = strides[-1]
        batch_width = int(np.ceil(batch_width/max_stride)*max_stride)
        batch_height = int(np.ceil(batch_height/max_stride)*max_stride)

        labels = []
        gtboxes = []
        batch_images = torch.zeros((batch_size, 3, batch_height, batch_width))
        input_size = np.array([batch_height, batch_width])
        for i in range(batch_size):
            img, label, boxes = batch[i]
            boxes = boxes.astype('float32')
            h, w = image_shapes[i]
            scale = image_scales[i]
            img = cv2.resize(img, (w, h))

            # OpenCV resize (W, H)
            boxes = boxes * scale
            if random.random() < 0.5:
                img, boxes = flip_img_boxes(img, boxes)

            # transform or data augmentation
            img = normalize_image(img)
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img)
            # assign anchors
            boxes = torch.Tensor(boxes)
            label, boxes = anchor_layer.assign(boxes, label, input_size=input_size,
                                               neg_thresh=config['negative_anchor_threshold'],
                                               pos_thresh=config['positive_anchor_threshold'])

            labels.append(label.unsqueeze(0))
            gtboxes.append(boxes.unsqueeze(0))
            # print(img.shape, batch_images.shape)
            batch_images[i, :, :h, :w] = img

        labels = torch.cat(labels, dim=0)
        gtboxes = torch.cat(gtboxes, dim=0)
        return batch_images, labels, gtboxes

    return collate_minibatch
