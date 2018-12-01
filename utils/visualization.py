"""

"""

import cv2
import json
import random
import argparse


def show_img(im_path, boxes):

    img = cv2.imread(im_path)
    for bb in boxes:
        if bb[4] < 0.3:
            continue
        img = cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 1)
        img = cv2.putText(img, '{}:{:.2f}'.format(bb[5], bb[4]), (int(bb[0]), int(bb[1])+10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 255, 0), 1)
    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dt', default='', type=str)
    args = parser.parse_args()
    with open(args.dt, 'r') as f:
        lines = f.readlines()
    lines = [json.loads(x.rstrip('\n')) for x in lines]
    random.shuffle(lines)
    dt = dict()
    for l in lines:
        name = l['image_id']
        res = l['result']
        _boxes = []
        for bb in res:
            _boxes.append(bb['bbox']+[bb['prob'], bb['class']])
        dt[name] = _boxes

    for k in dt.keys():
        show_img('/public_datasets/SynthText/'+k, dt[k])


if __name__ == '__main__':

    main()
