"""

"""
import json
import argparse
import numpy as np
from IPython import embed


def calculate_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def eval_ap(predict_path, gt_path, iou_thresh):

    with open(predict_path, 'r') as f:
        lines = f.readlines()
        predictions = [json.loads(x.rstrip('\n')) for x in lines]

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        gt = [json.loads(x.rstrip('\n')) for x in lines]

    predict_boxes = []
    for p in predictions:
        im_name = p['image_id']
        boxes = p['result']
        for bb in boxes:
            bb['im_name'] = im_name
            predict_boxes.append(bb)

    gt_boxes = dict()
    npos = 0
    for g in gt:
        gt_boxes[g['im_name']] = {'box': np.array(g['gtboxes']),
                                  'flag': np.zeros(len(g['gtboxes']), dtype=int)}
        npos += len(g['gtboxes'])

    # sort
    predict_boxes = sorted(predict_boxes, key=lambda x: x['prob'], reverse=True)
    tp = np.zeros(len(predict_boxes))
    fp = np.zeros(len(predict_boxes))
    for i in range(len(predict_boxes)):
        box = predict_boxes[i]
        im_name = box['im_name']
        _gt_boxes = gt_boxes[im_name]['box']
        bb = box['bbox']
        bb = np.array(bb)

        if len(_gt_boxes) > 0:

            ixmin = np.maximum(_gt_boxes[:, 0], bb[0])
            iymin = np.maximum(_gt_boxes[:, 1], bb[1])
            ixmax = np.minimum(_gt_boxes[:, 2], bb[2])
            iymax = np.minimum(_gt_boxes[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (_gt_boxes[:, 2] - _gt_boxes[:, 0] + 1.) *
                   (_gt_boxes[:, 3] - _gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > iou_thresh:
                if gt_boxes[im_name]['flag'][jmax] > 0:
                    fp[i] = 1
                else:
                    tp[i] = 1
                    gt_boxes[im_name]['flag'][jmax] = 1
            else:
                fp[i] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = calculate_ap(recall, precision)

    return ap


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict', type=str, default='', required=True)
    parser.add_argument('-g', '--gt', type=str, default='/public_datasets/SynthText/val.odgt')
    parser.add_argument('-t', '--thresh', type=float, default=0.5)

    args = parser.parse_args()

    ap = eval_ap(args.predict, args.gt, args.thresh)

    print("eval finished, ap={:.3f}".format(ap))


if __name__ == '__main__':

    main()





















