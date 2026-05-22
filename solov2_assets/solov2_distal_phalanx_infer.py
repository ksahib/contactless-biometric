default_scope = "mmdet"

metainfo = {
    "classes": ("distal_phalanx",),
    "palette": [(220, 20, 60)],
}

model = dict(
    type="SOLOv2",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=0,
        num_outs=5,
    ),
    mask_head=dict(
        type="SOLOV2Head",
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=128,
            mask_stride=4,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        ),
        loss_mask=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
    ),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel="gaussian",
        sigma=2.0,
        max_per_img=100,
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    name="visualizer",
    vis_backends=vis_backends,
)
