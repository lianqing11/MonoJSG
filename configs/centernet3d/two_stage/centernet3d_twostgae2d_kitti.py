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
        conv_channel=64,),
    two_stage_head=dict(
        type='TwoStage2DHead',
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),)
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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomRandomFlip3Dv2', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes'])]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
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

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project="det3d"))
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
