"""
ICDAR2015 for Text Detection
"""

import os
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from datasets.utils import normalize_image, get_im_scale


CLASSES = ('text',)
NUM_CLASSES = 2


class ICDAR15(Dataset):

    def __init__(self, dataroot, config, imageset='train'):
        assert imageset == 'train' or imageset == 'val' or imageset == 'all'
        self._imageset = imageset
        self._annotation_file = os.path.join(dataroot, '{}.odgt'.format(imageset))
        self._base_dir = os.path.join(dataroot, '{}_images'.format(imageset))
        self.name = 'ICDAR15'
        self.config = config
        self.annotations = self._read_annotations()

    def _read_annotations(self):
        # im path -> annotations
        with open(self._annotation_file, 'r') as f:
            lines = f.readlines()
        lines = list(map(lambda x: json.loads(x.rstrip('\n')), lines))
        return lines

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        im_name = annotation['im_name']
        gt_boxes = annotation['gtboxes']
        try:
            img = Image.open(os.path.join(self._base_dir, im_name))
        except OSError as e:
            idx = random.randint(0, len(self))
            annotation = self.annotations[idx]
            im_name = annotation['im_name']
            gt_boxes = annotation['gtboxes']
            img = Image.open(os.path.join(self._base_dir, im_name))

        if self._imageset == 'val':
            # testing or validation mode, original scale
            img = np.array(img).astype('float32')
            h, w = img.shape[:2]
            resize_h, resize_w, scale = get_im_scale(h, w, target_size=self.config['test_image_size'][0],
                                                     max_size=self.config['test_max_image_size'])
            img = cv2.resize(img, (resize_w, resize_h))
            img = normalize_image(img)
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img)
            return img, im_name, scale, (h, w)

        img = np.array(img).astype('float32')
        labels = np.ones(len(gt_boxes), dtype=np.int32)
        labels = torch.LongTensor(labels)
        boxes = np.array(gt_boxes, dtype=np.float32)
        # C, H, W

        return img, labels, boxes


