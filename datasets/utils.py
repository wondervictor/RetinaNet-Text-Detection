"""

Dataset utils

"""
import cv2
import numpy as np
from PIL import Image


def flip_pil_img_and_boxes(img, boxes=None):
    """ Flip PIL Images and Boxes
    Args:
        img: PIL Image
        boxes: [N, 4]
    """
    assert isinstance(img, Image.Image), "img should be PIL.Image"
    w, h = img.size
    flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes is not None:
        flip_boxes = boxes.copy()
        flip_boxes[:, 0] = w - boxes[:, 2] - 1
        flip_boxes[:, 2] = w - boxes[:, 0] - 1
        return flip_img, flip_boxes
    else:
        return flip_img


def flip_img_boxes(img, boxes=None):

    h, w, c = img.shape
    flip_img = cv2.flip(img, 1)
    if boxes is not None:
        flip_boxes = boxes.copy()
        for i in range(flip_boxes.shape[0]):
            flip_boxes[i, 0] = w - boxes[i, 2] - 1
            flip_boxes[i, 2] = w - boxes[i, 0] - 1
        return flip_img, flip_boxes
    else:
        return flip_img


def normalize_image(img):
    img = img / 255.0
    mean = np.array([.485, .456, .406])
    std = np.array([.229, .224, .225])
    img = (img - mean) / std
    return img


def get_im_scale(h, w, target_size, max_size):
    img_min_size = min(h, w)
    img_max_size = max(h, w)
    scale = target_size / img_min_size
    if scale * img_max_size > max_size:
        scale = max_size / img_max_size

    return int(round(h*scale)), int(round(w*scale)), scale
