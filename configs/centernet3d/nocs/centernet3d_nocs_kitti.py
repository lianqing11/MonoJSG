plugin=True
plugin_dir='det3d/'

#
model = dict(
    type='CenterNet3D',
    backbone=dict(
        type='DCNDLA',),
    neck=dict(
        type="IdentityNeck",),
    bbox_head=dict(
        type='CenterNet3DHead',
        num_classes=3,
        input_channel=64,
        conv_channel=256,),
    nocs_head=dict(
        type="NocsHead",
        loss_nocs=dict(
            type="L1Loss",
            loss_weight=10,),
        loss_unsupervised_nocs=dict(type="UncertaintyL1Loss"),
        nocs_coder=dict(type="NOCSCoder"),
        normalize_nocs=True,
        tanh_activate=False),
)


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'CustomMonoKittiDataset'
data_root = 'data/kitti/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

train_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_bbox=True,),
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='CustomRandomFlip3Dv2', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='ProjectLidar2Image'),
    dict(type='GenerateNocs', box_outside_range=0.1),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
                                       'dense_depth', 'object_coordinate',
                                       'valid_coordinate_mask', 'dense_location',
                                       'dense_dimension', 'dense_yaw', 'foreground_mask'])]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            remove_hard_instance_level=1,
            test_mode=False)),
    val=dict(
        type='ConcatDataset',
        datasets=[dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_train.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_val.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True),],
        separate_eval=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True))

optimizer = dict(
    type='AdamW',
    lr=0.0003,
    weight_decay=0.00001,)
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
lr_config = dict(policy='step', step=[16, 18])
total_epochs = 20

checkpoint_config = dict(interval=1, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project="monojsg"))
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
