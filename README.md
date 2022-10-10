## MonoJSG
MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection

## Quick Start
### Installation
**Step 0.** Follow the [readme](./software/mmdet3d/README.md) to setup the environment for MMDet3D.
**Step 1.** Setup the KITTI dataset and preprocess the data as in MMDet3D.
**Step 2.** Build [DCNv2](./det3d/models/backbones/DCNv2_t18/) by `bash make.sh`.

### Training
**Step 0.** Pretrain the detection network with NOCS 
```
bash tools/dist_train.sh configs/centernet3d/nocs/centernet3d_nocs_kitti.py 1
```
**Step 1.** Finetune the pretrained model with MonoJSG.
```
bash tools/dist_train.sh configs/centernet3d/two_stages/centernet3d_monojsg_kitti.py 1 --cfg-options load_from=$pretrained_model
```