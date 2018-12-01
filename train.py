"""

Training RetinaNet


"""
import os
import tqdm
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models import retina
from datasets import synthtext, icdar15, minibatch
from torch.utils.data import DataLoader
from lib.det_ops.loss import SigmoidFocalLoss, SmoothL1Loss
from IPython import embed
import tensorboardX
from utils import logger
from cfgs import config as cfg


def initialize(config, args):

    logdir = config['logdir']
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(os.path.join(logdir, args.experiment)):
        os.mkdir(os.path.join(logdir, args.experiment))

    model_dump_dir = os.path.join(logdir, args.experiment, 'model_dump')
    tb_dump = os.path.join(logdir, args.experiment, 'tb_dump')

    if not os.path.exists(model_dump_dir):
        os.mkdir(model_dump_dir)

    if not os.path.exists(tb_dump):
        os.mkdir(tb_dump)

    config['tb_dump_dir'] = tb_dump
    config['model_dump_dir'] = model_dump_dir


def learning_rate_decay(optimizer, step, config):
    base_lr = config['base_lr']
    lr = base_lr
    if step >= config['lr_decay'][0]:
        lr = base_lr * 0.1
    if step >= config['lr_decay'][0]:
        lr = base_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, config):
    anchor_scales = config['anchor_sizes']
    anchor_apsect_ratios = config['anchor_aspect_ratios']
    num_anchors = len(anchor_scales) * len(anchor_apsect_ratios)

    model = retina.RetinaNet(config['num_classes'], num_anchors, config['basemodel_path']).cuda()
    model = nn.DataParallel(model, device_ids=list(range(args.device)))

    if args.dataset == 'SynthText':
        train_dataset = synthtext.SynthText(dataroot=config['data_dir'], imageset=args.imageset, config=config)
    elif args.dataset == 'ICDAR':
        train_dataset = icdar15.ICDAR15(dataroot=config['data_dir'], imageset=args.imageset, config=config)
    else:
        raise NotImplemented()

    collate_minibatch = minibatch.create_minibatch_func(config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size*args.device,
        shuffle=True,
        num_workers=config['workers'],
        collate_fn=collate_minibatch
    )

    writer = tensorboardX.SummaryWriter(config['tb_dump_dir'])
    # torch model

    optimizer = optim.SGD(lr=config['base_lr'], params=model.parameters(),
                          weight_decay=config['weight_decay'], momentum=0.9)

    cls_criterion = SigmoidFocalLoss().cuda()
    box_criterion = SmoothL1Loss().cuda()

    start_epoch = 0
    global_step = 0

    # Load state dict from saved model
    if len(args.continue_path) > 0:
        model_state, optimizer_state, epoch, step = logger.load_checkpoints(args.continue_path)
        model.module.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        global_step = step+1
        start_epoch = epoch + 1

    for epoch in range(start_epoch, config['epochs']):
        losses = []
        data_iter = iter(train_loader)
        pbar = tqdm.tqdm(range(len(train_loader)))
        for i in pbar:
            img, labels, boxes = next(data_iter)
            img = img.cuda()
            labels = labels.long().cuda()
            boxes = boxes.cuda()
            cls_outputs, bbox_outputs = model(img)
            cls_loss = cls_criterion(cls_outputs, labels)
            box_loss = box_criterion(bbox_outputs, boxes, labels)
            loss = cls_loss + box_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/box_loss', box_loss.item(), global_step)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_step)
            global_step += 1
            pbar.set_description('e:{} i:{} loss:{:.3f} cls_loss:{:.3f} box_loss:{:.3f}'.format(
                epoch, i + 1, loss.item(), cls_loss.item(), box_loss.item()
            ))
            losses.append(loss.item())

            # learning rate decay
            learning_rate_decay(optimizer, global_step, config)

        print("e:{} loss: {}".format(epoch, np.mean(losses)))
        logger.save_checkpoints(model.module, optimizer, epoch, global_step,
                                path=os.path.join(config['model_dump_dir'],
                                                  'epoch-{}-iter-{}.pth'.format(epoch, global_step)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=1, help='training with ? GPUs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size per GPU')
    parser.add_argument('-c', '--continue_path', type=str, default='', help='continue model parameters')
    parser.add_argument('-e', '--experiment', type=str, default='synth_baseline',
                        help='experiment name, correspond to `config.py`')
    parser.add_argument('-ds', '--dataset', type=str, default='SynthText', help='dataset')

    _args = parser.parse_args()
    config = cfg.config[_args.experiment]
    train(_args, config)
