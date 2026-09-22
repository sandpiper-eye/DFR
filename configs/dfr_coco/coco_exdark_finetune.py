_base_ = [
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.dfr_yolo',
        'mmdet.utils.dfr_modules',
        'mmdet.models.losses.dfr_losses',
        'mmdet.engine.hooks.yolox_mode_switch_hook'
    ],
    allow_failed_imports=False
)

# --------------------------------------------------
# Load Stage 2 checkpoint
# --------------------------------------------------
# This loads ALL weights: backbone, neck, bbox_head, attribute_encoder,
# head_film_modules, feature_reorganization, etc.
load_from = 'work_dirs/dfr_coco/coco_exdark_generation/no1_10.pth'

# --------------------------------------------------
# Model config
# --------------------------------------------------
# CRITICAL: Model type must be DFRYOLO (not YOLOV3) to retain
# attribute_encoder and head_film_modules.
model = dict(
    type='DFRYOLO',

    # --- Stage control ---
    training_stage='joint',          # Unfreeze all detection modules
    freeze_backbone_bn_layers=0,     # Not used in joint, set to 0
    freeze_neck_bn=False,            # Not used in joint

    # --- DFR module switches ---
    enable_dfr=False,                # Disable DFR loss computation
    enable_reconstruction=False,     # Disable AdaIN feature reorganization
    enable_attribute_mixup=False,    # Disable attribute mixup
    enable_head_film=True,           # KEEP FiLM active for domain adaptation

    # DFR loss weights (not used when enable_dfr=False, kept for safety)
    dfr_loss_weight=0.0,
    dfr_recon_loss_weight=0.0,
    dfr_cross_loss_weight=0.0,
    dfr_content_loss_weight=0.0,
    dfr_ortho_loss_weight=0.0,
    dfr_attr_sep_loss_weight=0.0,
    attr_monitor_interval=100,

    # --- Data preprocessor (same as Stage 1/2) ---
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32),

    # --- Backbone (same as Stage 1/2) ---
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=None),

    # --- Attribute Encoder (REQUIRED by FiLM) ---
    # Must be structurally identical to Stage 2 for weight loading.
    # Freeze during Stage 3 to preserve domain separation.
    attribute_encoder=dict(
        type='AttributeEncoder',
        out_dim=256,
        normalize_output=True,
        norm_scale=5.0,
    ),

    # --- Neck (same as Stage 1/2) ---
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),

    # --- Feature Reorganization (kept for weight loading, unused) ---
    reorganization=dict(
        type='FeatureReorganizationModule',
        content_dim=[512, 256, 128],
        style_dim=256,
        num_levels=3,
        gamma_gate_init=1.0,
    ),

    # --- BBox Head (ExDark: 12 classes) ---
    # NOTE: When loading from Stage 2 (20 classes), the classification layer
    # weights have shape mismatch. MMDet skips mismatched layers with a warning.
    # This is expected and safe.
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=12,                # ExDark has 12 classes
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),

    # --- Feature Consistency Loss (kept for weight loading, unused) ---
    feature_consistency_loss=dict(
        type='FeatureConsistencyLoss',
        perception_model='resnet18',
        perception_layer='layer2',
        content_channels=512,
        target_channels=128,
    ),

    # --- Training / Test config ---
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_thr=0.45),
        max_per_img=100),
)

# --------------------------------------------------
# Dataset: ExDark (COCO format, 12 classes)
# --------------------------------------------------
dataset_type = 'CocoDataset'
data_root = 'DFRdata/exdarkcoco/'

metainfo = dict(
    classes=('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
             'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'),
    palette=[
        '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
        '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
    ]
)

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotations/val.json',
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

# --------------------------------------------------
# Evaluator: COCO metric
# --------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotations/val.json',
    metric='bbox',
    classwise=True, 
    format_only=False,
)
test_evaluator = val_evaluator

# --------------------------------------------------
# Optimizer: Layer-wise LR for fine-tuning
# --------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.': dict(lr_mult=1.0),
            'neck.': dict(lr_mult=1.0),
            'bbox_head.': dict(lr_mult=1.0),
            'attribute_encoder.': dict(lr_mult=0.0),       # FREEZE
            'head_film_modules.': dict(lr_mult=1.0),        # Full speed
            'feature_reorganization.': dict(lr_mult=0.0),   # Unused
            'loss_': dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=2,
)

# --------------------------------------------------
# Training schedule
# --------------------------------------------------
max_epochs = 75
num_last_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[50, 80],
        gamma=0.1),
]

# --------------------------------------------------
# Hooks
# --------------------------------------------------
custom_hooks = [
    #dict(
    #    type='YOLOXModeSwitchHook',
    #    num_last_epochs=num_last_epochs,
    #    priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
]
