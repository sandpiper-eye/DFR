_base_ = '../_base_/default_runtime.py'

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True, 
    broadcast_buffers=False,      
)

# load_from = 'work_dirs/dfr_yolo/dfr_yolo_detection/epoch_81.pth'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.dfr_yolo',
        'mmdet.utils.dfr_modules',
        'mmdet.models.losses.dfr_losses',
        'mmdet.datasets.lowlight',  
        'mmdet.engine.hooks.deepspeed_checkpoint_hook',
        'mmdet.utils.physical_degradation',
        'mmdet.datasets.samplers.ratio_sampler'
    ],
    allow_failed_imports=False
)
# model settings
model = dict(
    type='DFRYOLO',
    training_stage = 'generation',
    enable_dfr = True,
    enable_reconstruction = True,
    enable_head_film = True,    
    dfr_loss_weight=0.1,
    dfr_recon_loss_weight=0.1,
    dfr_cross_loss_weight=0.05,
    dfr_content_loss_weight=0.0,   
    dfr_ortho_loss_weight=0.1,
    dfr_attr_sep_loss_weight=0.15, 
    use_physical_degradation=True,
    dfr_equivariant_loss_weight=1.0,  
    attr_monitor_interval=100, 
    stage2_phase_iters=(5600, 12800),
    use_progressive_asymmetric_degradation=True,
    freeze_backbone_bn_layers = 3,  
    freeze_neck_bn = True,          
    init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/dfr_coco/coco_detection/epoch_57.pth',
            #revise_keys=[(r'^module\.', '')],  
            #strict=False,  
        ),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(type='Darknet', depth=53, out_indices=(3, 4, 5), init_cfg=None),
    attribute_encoder=dict(
        type='AttributeEncoder',
        out_dim=256,
        normalize_output=True,     
        norm_scale=5.0,            
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    reorganization=dict(
        type='FeatureReorganizationModule',
        content_dim = [512, 256, 128],
        style_dim = 256,
        num_levels = 3,
        gamma_gate_init = 0.0,  # Stage 2 initial: identity mapping, ramp up later
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=12,
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
    feature_consistency_loss=dict(
        type='FeatureConsistencyLoss',
        perception_model='resnet18',
        perception_layer='layer2',
        content_channels=512,
        target_channels=128  
    ),
    train_cfg = dict(
        assigner=dict(
            type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_thr=0.45),
        max_per_img=100))
# training and testing settings

# dataset settings
dataset_type = 'CocoDataset'
data_root_coco = 'DFRdata/coco12c/'
data_root_lowlight = 'DFRdata/exdarkcoco/train/images/'
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
    #dict(type='PhotoMetricDistortion'),
    dict(type='Resize', scale=(320,608), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            (dict(type='Resize', scale=640, keep_ratio=True),
             dict(type='Resize', scale=800, keep_ratio=True)),
            (dict(type='RandomFlip', prob=1.),
             dict(type='RandomFlip', prob=0.)),
        ],
        aggregate=True)
]

unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 608), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='RatioSampler', batch_size=16, source_ratio=[1, 1], shuffle=True, num_batches_per_epoch=800),  # P1: guarantee 2 coco + 2 ExDark per batch
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset_type', 'classes', 'palette'],
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root_coco,
                metainfo=metainfo,
                ann_file='annotations/train.json',
                data_prefix=dict(img='images/train/'),
                pipeline=train_pipeline
            ),
            dict(
                type='LowlightDataset',
                data_root=data_root_lowlight,
                pipeline=unsup_pipeline
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_coco,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 640), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]))

test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),  
    paramwise_cfg=dict(
        custom_keys={
            # P0-1
            'loss_ortho.':    dict(lr_mult=0.0, decay_mult=0.0),
            'loss_content.':  dict(lr_mult=0.0, decay_mult=0.0),
            'loss_attr_sep.': dict(lr_mult=0.0, decay_mult=0.0),
            'loss_equivariant.': dict(lr_mult=1.0),  # param_proj
            'feature_consistency_loss.projector.': dict(lr_mult=1.0),
            'backbone.': dict(lr_mult=0.0),
            'neck.': dict(lr_mult=0.0),
            'bbox_head.': dict(lr_mult=0.0),
            'attribute_encoder.': dict(lr_mult=1.0),
            'feature_reorganization.': dict(lr_mult=1.0),
            'head_film_modules.': dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),  
    accumulative_counts=2  
)

max_epochs = 25
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_coco + 'annotations/val.json',
    metric='bbox',
    classwise=True,
    format_only=False
)
test_evaluator = val_evaluator

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=1.0)
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
]

default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'), 
)

