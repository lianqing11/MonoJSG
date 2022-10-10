nohup bash debug.sh 0 bash tools/dist_test.sh configs/centernet3d/two_stage/centernet3d_nocs_lidar_kitti.py \
    ../det3d/work_dirs/centernet3d_nocs_kitti/1006_1802-/epoch_20.pth 1 --eval bbox >> ./work_dirs/nocs_kitti_1006_1802.log
nohup bash debug.sh 1 bash tools/dist_test.sh configs/centernet3d/two_stage/centernet3d_nocs_lidar_kitti.py \
    ../det3d/work_dirs/centernet3d_nocs_kitti/1006_1807-model.nocs_head.nocs_coder.with_yaw\:False-/epoch_20.pth 1 --eval bbox --cfg-options model.nocs_head.nocs_coder.with_yaw=False >> ./work_dirs/nocs_kitti_1006_1807_with_yaw=False.log 
nohup bash debug.sh 0 bash tools/dist_test.sh configs/centernet3d/two_stage/centernet3d_nocs_lidar_kitti.py \
    ../det3d/work_dirs/centernet3d_nocs_kitti/11006_1812-models.nocs_head.loss_nocs.loss_weight:10.0-/epoch_20.pth 1 --eval bbox >> ./work_dirs/nocs_kitti_1006_1812_loss_weight_10.log
nohup bash debug.sh 0 bash tools/dist_test.sh configs/centernet3d/two_stage/centernet3d_nocs_lidar_kitti.py \
    ../det3d/work_dirs/centernet3d_nocs_kitti/1006_1802-/epoch_20.pth 1 --eval bbox >> ./work_dirs/nocs_kitti_1006_1802.log