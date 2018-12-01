"""
Config File
"""


config = {

    "synth_baseline": {
        # lr and general config
        'base_lr': 1e-2,
        "lr_decay": [60000, 80000],
        "workers": 8,
        "num_classes": 21,
        "weight_decay": 1e-4,
        "epochs": 200,

        "basemodel_path": '/home/tianhengcheng/.torch/models/resnet50-19c8e357.pth',
        "data_dir": "/public_datasets/SynthText",

        # anchor config
        "positive_anchor_threshold": 0.5,
        "negative_anchor_threshold": 0.4,
        "anchor_sizes": [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        "aspect_ratios": [1, 3, 5],
        "anchor_areas": [32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 512 ** 2],
        "strides": [8, 16, 32, 64, 128],
        "base_size": 8,

        # dataset
        "image_scales": [600],
        "max_image_size": 1000,

        # test config
        "pre_nms_boxes": 1000,
        "test_nms": 0.5,
        "test_max_boxes": 300,
        "cls_thresh": 0.05,

        # log
        "logdir": "log",
        "tb_dump_dir": "",
        "model_dump_dir": "",
    },

    "icdar_baseline": {
        # lr and general config
        'base_lr': 1e-2,
        "lr_decay": [60000, 80000],
        "workers": 8,
        "num_classes": 21,
        "weight_decay": 1e-4,
        "epochs": 200,

        "basemodel_path": '/home/tianhengcheng/.torch/models/resnet50-19c8e357.pth',
        "data_dir": "/public_datasets/Text/icdar2015/",

        # anchor config
        "positive_anchor_threshold": 0.5,
        "negative_anchor_threshold": 0.4,
        "anchor_sizes": [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        "aspect_ratios": [1, 3, 5],
        "anchor_areas": [32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 512 ** 2],
        "strides": [8, 16, 32, 64, 128],
        "base_size": 8,

        # dataset
        "image_scales": [600],
        "max_image_size": 1000,

        # test config
        "pre_nms_boxes": 1000,
        "test_nms": 0.5,
        "test_max_boxes": 300,
        "cls_thresh": 0.05,

        # log
        "logdir": "log",
        "tb_dump_dir": "",
        "model_dump_dir": "",
    }

}
