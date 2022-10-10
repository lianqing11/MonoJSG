import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

import torch.nn.functional as F
from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr)
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models.builder import HEADS, build_loss

from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
from .centernet_utils import CenterNetHeatMap, CenterNetDecoder, decode_dimension, gather_feature
from det3d.core.bbox.util import alpha_to_ry, points_img2cam, points_img2cam_batch, \
                    projected_gravity_center, projected_2d_box, bbox_alpha

from inplace_abn import InPlaceABN
# from .loss_utils import reg_l1_loss, FocalLoss, depth_uncertainty_loss, BinRotLoss
import math

from .centernet3d_head import SingleHead
from det3d.core.visualizer.utils import draw_nocs_img

from mmcv import ops
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
import numpy as np
import os.path as osp
from PIL import Image

from mmdet3d.core.bbox.coders import build_bbox_coder


@HEADS.register_module()
class NocsHead(nn.Module):
    """

        Head for predict nocs
    """

    def __init__(self,
                input_channel=64,
                conv_channel=256,
                stride=4,
                normalize_nocs=False,
                with_seg_mask_head=True,
                nocs_coder=dict(type="NOCSCoder"),
                loss_nocs=None,
                loss_unsupervised_nocs=None,
                vis_nocs=False,
                tanh_activate = False,
                vis_nocs_config = dict(
                    type="RoIAlign", output_size=114,
                    spatial_scale=1, sampling_ratio=0, use_torchvision=True)):
        super().__init__()
        self.normalize_nocs = normalize_nocs
        nocs_output_channel = 3
        self.with_seg_mask_head = with_seg_mask_head
        if with_seg_mask_head is True:
            self.seg_mask_head = SingleHead(input_channel, conv_channel, 1)
        else:
            self.seg_mask_head = None

        self.stride = stride
        self.nocs_coder = build_bbox_coder(nocs_coder)

        self.loss_nocs = build_loss(loss_nocs)
        if loss_unsupervised_nocs is not None:
            self.loss_unsupervised_nocs = build_loss(loss_unsupervised_nocs)
            nocs_output_channel+=1
        self.nocs_head = SingleHead(input_channel,
                                    conv_channel,
                                    nocs_output_channel)
        self.vis_nocs = vis_nocs
        if self.vis_nocs:
            layer_cls = getattr(ops, vis_nocs_config['type'])
            vis_nocs_config.pop("type")
            # self.vis_nocs_co
            self.vis_nocs_roi_layer = layer_cls(**vis_nocs_config)

        self.tanh_activate = tanh_activate

    @property
    def with_loss_unsupervised_nocs(self):
        return hasattr(self, 'loss_unsupervised_nocs') \
                and self.loss_unsupervised_nocs is not None



    def forward(self, x):
        if isinstance(x, list):
            multiscale_features = x
            x = x[-1]

        nocs = self.nocs_head(x)
        if self.with_loss_unsupervised_nocs:
            nocs, nocs_uncertainty = nocs[:,:3], nocs[:,3:4]
        if self.tanh_activate:
            nocs = F.tanh(nocs)
        pred = {"nocs": nocs}

        if self.with_loss_unsupervised_nocs:
            pred['nocs_uncertainty'] = nocs_uncertainty

        if self.with_seg_mask_head:
            seg_mask = self.seg_mask_head(x)
            pred["seg_mask"] = seg_mask

        return pred

    def forward_train(self, x, input_metas,
                     object_coordinate, valid_coordinate_mask,
                     dense_location, dense_dimension, dense_yaw,
                     foreground_mask, **kwargs):
        preds = self(x)
        losses = self.loss(preds, input_metas, object_coordinate,
                          valid_coordinate_mask, dense_location,
                          dense_dimension, dense_yaw,
                          foreground_mask)

        return losses, preds


    def vis_nocs_func(self, boxes, nocs_dict, img_idx, input_meta,
                                    intrinsic, extrinsic,
                                    scores=None, scores_thr=0.01):
        # 1. get the 2d boxes
        if scores is not None:
            mask = scores > scores_thr
            if mask.sum() == 0:
                return None
        else:
            mask = boxes.tensor.new_ones(len(boxes)).bool()
        boxes.tensor = boxes.tensor[mask]
        nocs = nocs_dict['nocs']
        if 'seg_mask' in nocs_dict:
            seg_mask = nocs_dict['seg_mask']
        else:
            seg_mask = None
        if isinstance(boxes, LiDARInstance3DBoxes):
            boxes = boxes.convert_to(Box3DMode.CAM, extrinsic.to(boxes.tensor.device))
        boxes_2d = projected_2d_box(boxes, torch.tensor(intrinsic).cuda(), img_idx.shape[2:])

        nocs = F.interpolate(nocs, img_idx.shape[2:], mode="bilinear")
        if seg_mask is not None:
            seg_mask = F.interpolate(seg_mask, img_idx.shape[2:], mode="nearest")

        boxes_2d_extend = torch.cat([
            boxes_2d.new_zeros((len(boxes_2d), 1)).float(),
            boxes_2d], dim=1)

        # nocs_extend = torch.cat([

        # ])
        extracted_nocs = self.vis_nocs_roi_layer(nocs, boxes_2d_extend)
        extracted_raw_img = self.vis_nocs_roi_layer( img_idx, boxes_2d_extend)
        if seg_mask is not None:
            extracted_seg_mask = self.vis_nocs_roi_layer(seg_mask, boxes_2d_extend)

        # 2. roi-align to select the nocs

        # 3. roi-align to select the raw img

        # 3. check if it is normalized

        if self.normalize_nocs is True:
            # dense_dimension = F.interpolate(dense_dimension[0].permute(0, 3, 1, 2), pred_nocs.shape[-2:], mode="nearest")
            # dense_yaw = F.interpolate(dense_yaw[0].permute(0, 3, 1, 2), pred_nocs.shape[-2:], mode="nearest")
            # consider how to generate dimension and yaw based on roi layer
            dim = boxes.dims
            # TODO modify the hard code here
            yaw = boxes.yaw
            dense_dim = dim.reshape(-1, 3, 1, 1)
            dense_dim = dense_dim.expand(-1, -1,  extracted_nocs.shape[2], extracted_nocs.shape[3])
            dense_yaw = yaw.reshape(-1, 1, 1, 1)
            dense_yaw = dense_yaw.expand([-1, -1, extracted_nocs.shape[2], extracted_nocs.shape[3]])
            extracted_nocs = self.nocs_coder.decode(extracted_nocs, dense_dim, dense_yaw)
            # raise NotImplementedError
        # gravity_center = boxes.center.unsqueeze(-1).unsqueeze(-1).clone()
        # gravity_center
        nocs_3d = extracted_nocs + boxes.gravity_center.unsqueeze(-1).unsqueeze(-1)

        # 4. add the nocs with the predicted center
        nocs_3d_shape = nocs_3d.shape
        nocs_3d = nocs_3d.permute(0, 2, 3, 1).reshape(-1, 3)
        nocs_3d_extend = torch.cat([
                    nocs_3d,
                    nocs_3d.new_ones(len(nocs_3d), 1).float()
            ], dim=1)

        # nocs_2d = nocs_3d_extend @ intrinsic.T
        nocs_2d = torch.matmul(nocs_3d_extend, torch.Tensor(intrinsic).cuda().T)[..., :3]
        #projected_img_to_2d

        nocs_2d[:,:2] /= nocs_2d[:,2:3]

        nocs_2d[:,0] /= img_idx.shape[-1]
        nocs_2d[:,1] /= img_idx.shape[-2]
        nocs_2d = nocs_2d*2 - 1
        nocs_2d = nocs_2d[..., :2]
        # nocs_2d

        # 5. project it to 2d location
        nocs_2d = nocs_2d.reshape(nocs_3d_shape[0], nocs_3d_shape[2], nocs_3d_shape[3], 2)
        # nocs_2d = nocs_2d.permute(0, 3, 1, 2)
        nocs_img = F.grid_sample(img_idx.expand(len(nocs_2d), -1, -1, -1), nocs_2d, mode="bilinear")

        # 7. visualize the nocs img and origin img
        if seg_mask is not None: # this means extracted_seg_mask is also not None
            extracted_seg_mask = extracted_seg_mask.expand(-1, 3, -1, -1)

            extracted_raw_img[extracted_seg_mask.sigmoid() < 0.1] = 0
            nocs_img[extracted_seg_mask.sigmoid() < 0.1] = 0
        draw_nocs_img(input_meta, extracted_raw_img, nocs_img, extracted_seg_mask[:,0:1].sigmoid())


        return None

    def loss(self, pred_dict, input_metas, object_coordinate, valid_coordinate_mask,
                    dense_location, dense_dimension, dense_yaw, foreground_mask):
        """

        """
        # FROM N H W C  to N C H W
        dense_dimension = dense_dimension[0].permute(0, 3, 1, 2)
        dense_yaw = dense_yaw[0].permute(0, 3, 1, 2)
        dense_location = dense_location[0].permute(0, 3, 1 ,2)
        gt_nocs = object_coordinate[0].permute(0,3,1,2 )
        valid_mask = valid_coordinate_mask[0].permute(0, 3, 1, 2)
        foreground_mask = foreground_mask[0].permute(0, 3, 1, 2)

        device = pred_dict["nocs"].device
        # NOTE !!! currently, the model only support mono setting.
        intrinsic = torch.tensor([input_meta['cam2img'][0] for input_meta in input_metas])
        intrinsic = intrinsic.to(device)

        extrinsic = torch.tensor([input_meta['lidar2cam'][0] for input_meta in input_metas])
        extrinsic = extrinsic.to(device)
        output_shape = pred_dict["nocs"].shape[2:]

        pred_nocs = pred_dict["nocs"]
        # from the downsample version to the origin img resolution version
        pred_nocs = F.interpolate(pred_nocs, dense_dimension.shape[2:4], mode="bilinear")
        # currently I support num of multi-view image is 1
        # mask = dense_depth[0]
        loss = {}

        if self.normalize_nocs == True:
            # denormalize with the dimension and yaw;
            pred_nocs = self.nocs_coder.decode(pred_nocs, dense_dimension, dense_yaw)

        # gt_nocs = F.interpolate(gt_nocs, pred_nocs.shape[-2:], mode="nearest")
        # valid_mask = F.interpolate(valid_mask, pred_nocs.shape[-2:], mode="nearest")
        # valid means there is point cloud in the projected img pixel.
        valid_mask = valid_mask.bool().expand(-1, 3, -1, -1)
        if valid_mask.sum() > 0:
            # check the loss function # l1 loss?
            loss["loss_nocs"] = self.loss_nocs(pred_nocs[valid_mask], gt_nocs[valid_mask])
        else:
            loss["loss_nocs"] = pred_nocs.new_zeros(1)

        if self.with_loss_unsupervised_nocs:
            # only consider the foreground pixel (the pixel inside the bbox)
            valid_mask = foreground_mask
            valid_mask = valid_mask != 0
            # dense_location =
            pred_nocs_foreground = pred_nocs.permute(0, 2, 3, 1).reshape(-1, 3)
            pred_nocs_foreground = pred_nocs_foreground[valid_mask.reshape(-1).bool()]
            dense_location = dense_location.permute(0, 2, 3, 1).reshape(-1, 3)

            # pre[valid_mask]
            # calculate the unsupervised loss
            pred_nocs_3d = pred_nocs_foreground + dense_location[valid_mask.reshape(-1).bool()]
            pred_nocs_3d = torch.cat([
                pred_nocs_3d,
                pred_nocs_3d.new_ones(pred_nocs_3d.shape[0], 1)
            ], dim=1)
            # TODO
            dense_intrinsic = intrinsic.reshape(-1, 1, 1, 4, 4,)
            dense_intrinsic = dense_intrinsic.expand(-1, valid_mask.shape[2], valid_mask.shape[3], -1, -1).reshape(-1, 4, 4)
            dense_intrinsic = dense_intrinsic[valid_mask.reshape(-1).bool()]
            # project the nocs + location to the image plane.
            pred_nocs_2d = torch.bmm(pred_nocs_3d.unsqueeze(1), dense_intrinsic.permute(0, 2, 1))
            pred_nocs_2d = pred_nocs_2d.squeeze(1)

            # pred_nocs_2d /= pred_nocs_2d[:,2:3]
            # pred_nocs_2d[..., :2] /= pred_nocs_2d[..., 2:3]
            pred_nocs_depth = pred_nocs_2d[:, 2]
            pred_nocs_2d = pred_nocs_2d[:,:2]

            # strange inplace error
            pred_nocs_2d = pred_nocs_2d.clone() / pred_nocs_depth.unsqueeze(1).clamp(min=1)
            gt_nocs_2d = torch.stack(torch.meshgrid(
                torch.arange(valid_mask.shape[3]),
                torch.arange(valid_mask.shape[2])), dim=-1).permute(1, 0, 2)
            gt_nocs_2d = gt_nocs_2d.to(device=pred_nocs.device)
            gt_nocs_2d = gt_nocs_2d.unsqueeze(0).expand(valid_mask.shape[0], -1, -1, -1).reshape(-1, 2)
            gt_nocs_2d = gt_nocs_2d[valid_mask.reshape(-1).bool()]
            gt_nocs_2d = gt_nocs_2d #self.stride
            # calculate the unsupervised uncertainty;
            uncertainty = pred_dict["nocs_uncertainty"]
            uncertainty = F.interpolate(uncertainty, gt_nocs.shape[2:], mode="bilinear")
            uncertainty = uncertainty.reshape(-1)[valid_mask.reshape(-1).bool()]

            loss["loss_unsupervised_nocs"] = self.loss_unsupervised_nocs(
                pred_nocs_2d.reshape(-1, 2), gt_nocs_2d, uncertainty)

        if self.with_seg_mask_head:
            # calculate tht segmentation loss
            gt_seg_mask = foreground_mask
            # save_seg_mask(gt_seg_mask, valid_mask, input_metas, "v1")

            # gt_seg_mask = F.interpolate(gt_seg_mask, pred_nocs.shape[-2:], mode="nearest").float()
            # save_seg_mask(gt_seg_mask, valid_mask, input_metas, "v2")

            valid_mask = gt_seg_mask!= -1
            pred_seg_mask = pred_dict["seg_mask"]
            pred_seg_mask = F.interpolate(pred_seg_mask, pred_nocs.shape[-2:], mode="bilinear")
            # use the focal loss 
            loss["loss_seg_nocs"] = py_sigmoid_focal_loss(
                    pred_seg_mask[valid_mask].reshape(-1),
                    gt_seg_mask[valid_mask].reshape(-1))


            seg_foreground_mean = pred_seg_mask.sigmoid()[gt_seg_mask==1].mean()
            seg_background_mean = pred_seg_mask.sigmoid()[gt_seg_mask==0].mean()
            loss["seg_foreground_mean"] = seg_foreground_mean
            loss["seg_background_mean"] = seg_background_mean
        return loss


def save_seg_mask(gt_seg_mask, valid_mask, input_metas, mode="v1"):
    for idx in range(len(input_metas)):
        fileid = osp.basename(input_metas[idx]['filename'])
        seg_mask_idx = gt_seg_mask[idx].cpu().permute(1, 2, 0).expand(-1, -1, 3)
        print((seg_mask_idx==1).sum())
        seg_mask_idx = seg_mask_idx.numpy()
        seg_mask_idx *= 255
        seg_mask_idx[seg_mask_idx==-255] = 122
        seg_mask_idx = Image.fromarray(seg_mask_idx.astype(np.uint8))
        seg_mask_idx.save(fileid+ mode + "foreground_mask.png")


@HEADS.register_module()
class RefineByNocsHead(nn.Module):

    def __init__(self,
                refine_type="lidar",
                refine_depth_range=[-7, 7],
                refine_depth_grid=0.1,
                nocs_coder=dict(type="NOCSCoder"),
                refine_roi_config= dict(
                    type="RoIAlign", output_size=114,
                    spatial_scale=1, sampling_ratio=0,
                    use_torchvision=True),
                seg_mask_threshold=0.5):
        super().__init__()
        self.refine_type = refine_type
        self.nocs_coder = build_bbox_coder(nocs_coder)
        self.refine_depth_range = refine_depth_range
        self.refine_depth_grid = refine_depth_grid
        layer_cls = getattr(ops, refine_roi_config['type'])
        refine_roi_config.pop('type')
        self.refine_roi_layer = layer_cls(**refine_roi_config)
        self.seg_mask_threshold = seg_mask_threshold

    def forward_train(self, **kwargs):
        # this is a post-processing module, which has not the training function.
        return {}, None

    def forward_test(self, bbox_list, preds,
                     img, img_metas,**kwargs):
        '''
            bbox_list: the bounding boxes output from centernet
            preds: pred feature map
            img:
            img_metas:
        '''
        normalize_nocs = preds["normalize_nocs"]

        intrinsic = torch.tensor(img_metas[0]['cam2img'][0]).to(img.device)
        extrinsic = torch.tensor(img_metas[0]['lidar2cam'][0]).to(img.device)

        refine_length = self.refine_depth_range[1] - self.refine_depth_range[0]
        depth_grid = torch.arange(refine_length / self.refine_depth_grid + 1) * self.refine_depth_grid
        depth_grid = depth_grid + self.refine_depth_range[0]
        depth_grid = depth_grid.to(img.device)
        # depth_grid = torch.arange(self.refine_depth_range[1] - self.refine)
        nocs = preds["nocs"]

        if "seg_mask" in preds:
            seg_mask = preds["seg_mask"]
        else:
            seg_mask = None

        det_bboxes, det_scores, det_labels = bbox_list[0]

        if isinstance(det_bboxes, LiDARInstance3DBoxes):
            det_bboxes = det_bboxes.convert_to(Box3DMode.CAM, extrinsic)

        boxes = det_bboxes
        boxes_2d = projected_2d_box(boxes, intrinsic, img.shape[-2:])
        depth = kwargs["dense_depth"][0]
        nocs = F.interpolate(nocs, img.shape[-2:], mode="bilinear")
        if seg_mask is not None:
            seg_mask = F.interpolate(seg_mask, img.shape[-2:], mode="bilinear")
        projected_center = projected_gravity_center(boxes, intrinsic)
        bbox_depth = boxes.center[:, -1]
        candidate_depth = bbox_depth.unsqueeze(-1).expand(-1, len(depth_grid))
        candidate_depth = candidate_depth.clone()
        candidate_depth += depth_grid.unsqueeze(0).expand(len(bbox_depth), -1).clone()
        projected_center = projected_center.unsqueeze(1).expand(-1, candidate_depth.shape[1], -1)
        N, K = projected_center.shape[0], projected_center.shape[1]
        candidate_center = points_img2cam(projected_center.reshape(-1, 2),
                                            candidate_depth.reshape(-1, 1), intrinsic)
        candidate_center = candidate_center.reshape(N, K, 3)
        if self.refine_type == "lidar":
            depth = kwargs["dense_depth"][0]
            depth = depth.permute(0, 3, 1, 2)
            for jdx in range(len(boxes_2d)):
                # select the nocs by 2D boxes
                x1, y1, x2, y2 = boxes_2d[jdx]
                if int(y2) - int(y1) == 0 or int(x2) - int(x1) == 0:
                    continue
                nocs_jdx = nocs[:,:,int(y1): int(y2),
                                    int(x1): int(x2)]

                dim = boxes.dims[jdx]
                yaw = boxes.yaw[jdx]
                if normalize_nocs is True:
                    dense_dim = dim.reshape(-1, 3, 1, 1)
                    dense_dim = dense_dim.expand(
                        -1, 3,  nocs_jdx.shape[2], nocs_jdx.shape[3])
                    dense_yaw = yaw.reshape(-1, 1, 1, 1)
                    dense_yaw = dense_yaw.expand(
                        [-1, -1, nocs_jdx.shape[2], nocs_jdx.shape[3]])
                    try:
                        nocs_jdx = self.nocs_coder.decode(nocs_jdx, dense_dim, dense_yaw)
                    except:
                        import pdb; pdb.set_trace()

                candidate_center_jdx = candidate_center[jdx]


                depth_jdx = depth[:,:,int(y1): int(y2),
                                            int(x1): int(x2)]
                depth_jdx = depth_jdx.expand(candidate_center_jdx.shape[0], -1, -1, -1)

                # select the gt depth by 2D boxes
                nocs_jdx = nocs_jdx.expand(candidate_center_jdx.shape[0], -1, -1, -1).clone()

                nocs_jdx += candidate_center_jdx.reshape(-1, 3, 1, 1)

                empty_mask = depth_jdx[:,-1] >= 1.0
                empty_mask = empty_mask.unsqueeze(1).expand(-1, 3, -1, -1)
                if seg_mask is not None:
                    seg_mask_jdx = seg_mask[:,:,int(y1): int(y2),
                                                int(x1): int(x2)]
                    # empty_mask *= seg_mask_jdx >0.1

                # nocs_jdx = nocs_jdx[]
                depth_alignment_loss = (nocs_jdx - depth_jdx).abs()
                depth_alignment_loss *= empty_mask
                depth_alignment_loss = depth_alignment_loss.sum([1, 2, 3])
                # depth_alignment_loss = depth
                depth_alignment_loss = depth_alignment_loss / empty_mask.sum([1, 2, 3]).clamp(min=0.1)
                min_loss_value, min_loss_idx  = depth_alignment_loss.min(0)
                optimal_center = candidate_center_jdx[min_loss_idx]

                optimal_center[1] += dim[1]/2.
                det_bboxes.tensor[jdx,:3] = optimal_center

                # generate the candidate
            det_bboxes = det_bboxes.convert_to(Box3DMode.LIDAR,
                                rt_mat = torch.inverse(extrinsic))

            # 6. get loss (ignore empty region)
            # 7. N argmin(k) + location

        elif self.refine_type == "stereo":
            left_img = img.squeeze(0)
            right_img = kwargs["right_img"].squeeze(0)

            # get the baseline
            # initialize in the bev setting?
            baseline = img_metas[0]["cam2img"][1][0,-1] - img_metas[0]["cam2img"][0][0, -1]
            focal_length = intrinsic[0][0]

            # get the grid # extracted the grid
            # # based on the grid and the baselines, get the target grid
            # generate the target image
            boxes_2d_extend = torch.cat([
                boxes_2d.new_zeros((len(boxes_2d), 1)).float(),
                boxes_2d], dim=1)
            extracted_nocs = self.refine_roi_layer(nocs, boxes_2d_extend)
            # check the shape
            if normalize_nocs is True:
                dim = boxes.dims
                yaw = boxes.yaw
                dense_dim = dim.reshape(-1, 3, 1, 1)
                dense_dim = dense_dim.expand(
                    -1, 3,  extracted_nocs.shape[2], extracted_nocs.shape[3])
                dense_yaw = yaw.reshape(-1, 1, 1, 1)
                dense_yaw = dense_yaw.expand(
                    [-1, -1, extracted_nocs.shape[2], extracted_nocs.shape[3]])
                extracted_nocs = self.nocs_coder.decode(extracted_nocs, dense_dim, dense_yaw)

            extracted_img = self.refine_roi_layer(left_img, boxes_2d_extend)

            if seg_mask is not None:
                extracted_seg_mask = self.refine_roi_layer(seg_mask, boxes_2d_extend)

            # get the depth of the nocs
            # projected the nocs to the target img

            # N, K, = projected_center.shape
            N, _, roi_h, roi_w = extracted_nocs.shape
            extracted_nocs = extracted_nocs.unsqueeze(1).expand(-1, K, -1, -1, -1)
            extracted_nocs = extracted_nocs.reshape(N*K, 3, roi_h, roi_w)
            candidate_center = candidate_center.reshape(N*K, 3, 1, 1)
            grid = torch.stack(torch.meshgrid(
                torch.arange(left_img.shape[-1]),
                torch.arange(left_img.shape[-2])), dim=-1).permute(1, 0, 2)

            grid = grid.unsqueeze(0).permute(0, 3, 1, 2).to(img.device)

            extracted_grid = self.refine_roi_layer(grid.float(), boxes_2d_extend)
            extracted_grid = extracted_grid.unsqueeze(1).expand(-1, K, -1, -1, -1)
            extracted_grid = extracted_grid.reshape(N*K,2, roi_h, roi_w)
            disparity = extracted_nocs + candidate_center
            disparity = baseline /  disparity[:,2:3].clamp(min=1.0)

            extracted_grid[:,0:1] += disparity
            extracted_grid[:,0] /= left_img.shape[-1]
            extracted_grid[:,1] /= left_img.shape[-2]

            extracted_grid = extracted_grid * 2 - 1
            right_img = right_img.expand(N*K, -1, -1, -1)
            extracted_grid = extracted_grid.permute(0, 2, 3, 1)
            # extracted_grid = extracted_grid -
            target_img = F.grid_sample(right_img, extracted_grid, mode="bilinear", padding_mode="zeros")
            extracted_img = extracted_img.unsqueeze(1).expand(-1, K, -1, -1, -1)
            extracted_img = extracted_img.reshape(N*K, 3, roi_h, roi_w)
            photometric_alignment_loss = (extracted_img - target_img).abs()

            if seg_mask is not None:
                extracted_seg_mask = extracted_seg_mask.unsqueeze(1).expand(-1, K, -1, -1, -1)
                extracted_seg_mask = extracted_seg_mask.reshape(N*K, 1, roi_h, roi_w)
                photometric_alignment_loss = photometric_alignment_loss * extracted_seg_mask
                photometric_alignment_loss = photometric_alignment_loss.mean([1])
                photometric_alignment_loss = photometric_alignment_loss.sum([1, 2]) / \
                        extracted_seg_mask.sum([1, 2, 3]).clamp(min=0.1)
            else:
                photometric_alignment_loss = photometric_alignment_loss.mean([1,2,3])

            optimal_value, optimal_idx = photometric_alignment_loss.reshape(N, K).min(1)

            candidate_center = candidate_center.reshape(N, K, 3)
            optimal_idx = optimal_idx.clamp(max=K-1)
            # print(candidate_center)
            # print(optimal_idx)
            # print(candidate_center.shape)
            candidate_center = [candidate_center[idx, jdx] for idx, jdx in enumerate(optimal_idx)]
            optimal_center = torch.stack(candidate_center, dim=0)
            # candidate_center = candidate_center[optimal_idx]
            optimal_center[:, 1] += det_bboxes.dims[:, 1]/2.
            det_bboxes.tensor[:,:3] = optimal_center

            # get the optimal center
            det_bboxes = det_bboxes.convert_to(Box3DMode.LIDAR,
                                rt_mat = torch.inverse(extrinsic))


        elif self.refine_type == "mono": # 2D-3D proejction
            img = img.squeeze(0)



            boxes_2d_extend = torch.cat([
                boxes_2d.new_zeros((len(boxes_2d), 1)).float(),
                boxes_2d], dim=1)
            extracted_nocs = self.refine_roi_layer(nocs, boxes_2d_extend)
            # check the shape
            if normalize_nocs is True:
                dim = boxes.dims
                yaw = boxes.yaw
                dense_dim = dim.reshape(-1, 3, 1, 1)
                dense_dim = dense_dim.expand(
                    -1, 3,  extracted_nocs.shape[2], extracted_nocs.shape[3])
                dense_yaw = yaw.reshape(-1, 1, 1, 1)
                dense_yaw = dense_yaw.expand(
                    [-1, -1, extracted_nocs.shape[2], extracted_nocs.shape[3]])
                extracted_nocs = self.nocs_coder.decode(extracted_nocs, dense_dim, dense_yaw)

            extracted_img = self.refine_roi_layer(img, boxes_2d_extend)

            if seg_mask is not None:
                extracted_seg_mask = self.refine_roi_layer(seg_mask, boxes_2d_extend)

            # get the depth of the nocs
            # projected the nocs to the target img

            # N, K, = projected_center.shape
            N, _, roi_h, roi_w = extracted_nocs.shape
            extracted_nocs = extracted_nocs.unsqueeze(1).expand(-1, K, -1, -1, -1)
            extracted_nocs = extracted_nocs.reshape(N*K, 3, roi_h, roi_w)
            candidate_center = candidate_center.reshape(N*K, 3, 1, 1)

            extracted_nocs = extracted_nocs + candidate_center
            extracted_nocs = extracted_nocs.permute(0, 2, 3, 1).reshape(-1, 3)
            extracted_nocs = torch.cat([
                        extracted_nocs,
                        extracted_nocs.new_ones(len(extracted_nocs), 1).float()
                ], dim=1)
            extracted_nocs_2d = \
                 torch.matmul(extracted_nocs, intrinsic.T)[..., :3]

            extracted_nocs_2d[:,:2] /= extracted_nocs_2d[:,2:3]

            extracted_nocs_2d[:,0] /= img.shape[-1]
            extracted_nocs_2d[:,1] /= img.shape[-2]

            extracted_nocs_2d = extracted_nocs_2d*2 - 1
            extracted_nocs_2d = extracted_nocs_2d[..., :2]
            extracted_nocs_2d = extracted_nocs_2d.reshape(-1, roi_h, roi_w, 2)


            img = img.expand(N*K, -1, -1, -1)

            target_img = F.grid_sample(img, extracted_nocs_2d, mode="bilinear", padding_mode="zeros")
            extracted_img = extracted_img.unsqueeze(1).expand(-1, K, -1, -1, -1)
            extracted_img = extracted_img.reshape(N*K, 3, roi_h, roi_w)
            photometric_alignment_loss = (extracted_img - target_img).abs()
            if seg_mask is not None:
                extracted_seg_mask = extracted_seg_mask.unsqueeze(1).expand(-1, K, -1, -1, -1)
                extracted_seg_mask = extracted_seg_mask.reshape(N*K, 1, roi_h, roi_w)
                photometric_alignment_loss = photometric_alignment_loss * extracted_seg_mask
                photometric_alignment_loss = photometric_alignment_loss.mean([1])
                photometric_alignment_loss = photometric_alignment_loss.sum([1, 2]) / \
                        extracted_seg_mask.sum([1, 2, 3]).clamp(min=0.1)
            else:
                photometric_alignment_loss = photometric_alignment_loss.mean([1,2,3])

            optimal_value, optimal_idx = photometric_alignment_loss.reshape(N, K).min(1)

            candidate_center = candidate_center.reshape(N, K, 3)
            optimal_idx = optimal_idx.clamp(max=K-1)
            # print(candidate_center)
            # print(optimal_idx)
            # print(candidate_center.shape)
            candidate_center = [candidate_center[idx, jdx] for idx, jdx in enumerate(optimal_idx)]


            optimal_center = torch.stack(candidate_center, dim=0)
            # candidate_center = candidate_center[optimal_idx]
            optimal_center[:, 1] += det_bboxes.dims[:, 1]/2.
            det_bboxes.tensor[:,:3] = optimal_center

            # get the optimal center
            det_bboxes = det_bboxes.convert_to(Box3DMode.LIDAR,
                                rt_mat = torch.inverse(extrinsic))



        # sample bbox based on depth range

        # refine the bounding box based on different module

        # get the box that achieves the minimum loss

        return [(det_bboxes, det_scores, det_labels)]

    def simple_test(self, bbox_list, preds,
                     img, img_metas,**kwargs):

        return self.forward_test(bbox_list, preds,
                                 img, img_metas, **kwargs)



