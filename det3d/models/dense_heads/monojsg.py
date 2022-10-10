import numpy as np
import torch
from torch import nn as nn

import torch.nn.functional as F

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS, build_loss

from mmdet3d.core.bbox.structures import Box3DMode
from det3d.core.bbox.util import points_img2cam,\
                        points_img2cam_batch, points_cam2img_batch, \
                        projected_gravity_center, \
                        projected_2d_box, points_cam2img_broadcast


from .two_stage_head import TwoStageHead
from mmcv import ops


from det3d.models.necks.identity_neck import BasicBlock3dV2
from mmdet3d.core.bbox.coders import build_bbox_coder


@HEADS.register_module()
class MonoJSGHead(TwoStageHead):
    """
    Two stage head with semantic and geometric cost volume:
        Given the predicted nocs, generate the cost volume based on the candidate depth

    """

    def __init__(self,
                 num_classes=3,
                 input_channels=64,
                 feat_channels=256,
                 box_code_size=3,
                 nocs_coder=dict(type="NOCSCoder"),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_reg=dict(
                     type='L1Loss', loss_weight=1.0),
                 iou3d_thresh=0.2,
                 iou2d_thresh=0.5,
                 stride=4,
                 depth_grid_num=18,
                 depth_grid_size=0.2,
                 foreground_threshold = 0.1,
                 network_type="v1",
                 roi_config = dict(
                     type="RoIAlign", output_size=14,
                     spatial_scale=0.25, sampling_ratio=0, use_torchvision=True)):
        super().__init__(num_classes,
                         box_code_size=3,
                         iou3d_thresh=iou3d_thresh,
                         iou2d_thresh=iou2d_thresh,)
        self.input_channels = input_channels
        self.feat_channels = feat_channels
        self.box_code_size = box_code_size

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        self.foreground_threshold = foreground_threshold
        self.nocs_coder = build_bbox_coder(nocs_coder)

        roi_name = getattr(ops, roi_config['type'])
        roi_config.pop('type')
        self.roi_layer = roi_name(**roi_config)
        self.stride = stride

        self.depth_grid_num = depth_grid_num
        self.depth_grid_size = depth_grid_size

        depth_grid = torch.arange(depth_grid_num) * depth_grid_size - \
            (depth_grid_num-1) /2* depth_grid_size
        depth_grid = depth_grid.reshape(1, -1)
        self.register_buffer("depth_grid", depth_grid)

        self._init_layers(network_type)



    def _init_layers(self, network_type):
        if network_type == "v1":
            self.refine_module = RefineCostNetV1(
                        self.input_channels,  self.feat_channels)
            # self.mlp_cls = _mlp_module(self.feat_channels * self.depth_grid_num, 1)
            # self.mlp_reg = _mlp_module(self.feat_channels, 1)

        elif network_type == "v2":
            raise NotImplementedError



    def forward(self, features):

        # assert input with 4D cost volume for the mlp features

        output = {}
        output["reg_depth"] = self.refine_module(features, self.depth_grid)
        # N, C, D = features.shape
        # # cls_features = features.reshape(-1, C*D)
        # # uncertainty_features = features.reshape(-1, C*D)
        # reg_features = features.clone().permute(0, 2, 1).reshape(-1, C)

        # # features = features.mean
        # output = {}

        # # output["cls"] = self.mlp_cls(cls_features)
        # # output["uncertainty"] = self.uncertainty_ref

        return output



    def loss(self, refine_preds, gt, bbox_list):

        # based of depth grid and bbox list
        # generate depth candidate

        # calculate l1 loss for refine preds

        # return losses["loss_reg"] =
        reg_mask = (gt["cls"] != self.num_classes) & (gt["cls"] != -1)

        losses = {}
        if reg_mask.sum() > 0:
            losses["loss_2stage_depth"] = self.loss_reg(
                    refine_preds["reg_depth"][reg_mask],
                    gt["reg"][:, 2:3].detach()[reg_mask])
        return losses



    def get_bbox_refine(self, refine_preds, bbox_list, img_metas):
        bsz = len(img_metas)

        pred_reg = refine_preds["reg_depth"].reshape(bsz, -1, 1)
        num_pred = pred_reg.shape[1]
        new_bbox_list = []
        for idx in range(bsz):
            img_meta = img_metas[idx]
            pred_bbox = bbox_list[idx][0]
            if self.refine_proposal is True:
                # pred_bbox.tensor
                # get the uv, z, w, h, l, roty
                extrinsic = img_metas[idx]["lidar2cam"][0]
                extrinsic = torch.tensor(extrinsic).to(pred_bbox.tensor.device)

                intrinsic = img_metas[idx]["cam2img"][0]
                intrinsic = torch.tensor(intrinsic).to(pred_bbox.tensor.device)

                pred_bbox_cam = pred_bbox.convert_to(Box3DMode.CAM, rt_mat=extrinsic)
                uv = projected_gravity_center(
                            pred_bbox_cam, extrinsic, img_meta['img_shape'][0])

                depth = pred_bbox_cam.depth + refine_preds["reg_depth"]
                dimension = pred_bbox_cam.tensor[:,3:6]
                location = points_img2cam(uv, depth, intrinsic)
                location[:, 1] += dimension[:, 1]/2.
                pred_bbox_cam.tensor[:,:3] = location
                pred_bbox = pred_bbox_cam.convert_to(Box3DMode.LIDAR, rt=torch.inverse(extrinsic))
                scores, clses = bbox_list[idx][1], bbox_list[idx][2]
                new_bbox_list.append( (pred_bbox, scores, clses))
            else:
                new_bbox_list.append((bbox_list[idx]))

        return new_bbox_list


    def select_features(self, bbox_list, features, img_metas, preds):
        '''
        Args:

        Output:
            Features with shape of NCDHW
        '''
        nocs = preds["nocs"]
        # preds["normalized_nocs"] =
        normalize_nocs = preds["normalize_nocs"]
        seg_mask = preds["seg_mask"] if "seg_mask" in preds else None


        # cost_volume_features = []
        # generate the 2D features based on roi aligns
        if isinstance(features, list):
            features = features[-1]
        N, C, H, W = features.shape

        bboxes2d, bboxes3d, bboxes3d_cam = \
            get2d_bboxes(bbox_list, img_metas, features.device, with_3d=True)

        bboxes2d = bboxes2d.detach()
        bboxes3d_cam = bboxes3d_cam.detach()


        # generate 2d grid for geometry features
        img_grid = torch.meshgrid([
                torch.arange(H * self.stride, device=features.device),
                torch.arange(W * self.stride, device=features.device)])
        img_grid = torch.stack(img_grid, dim=-1).float()
        # from pytorch coord to nature coord (y, x) -> (x, y) with C, H, W
        img_grid = torch.flip(img_grid, dims=[-1]).permute(2, 0, 1)
        img_grid = img_grid.unsqueeze(0).expand(len(img_metas), -1, -1, -1)
        roi_features = self.roi_layer(features, bboxes2d)
        roi_grid = self.roi_layer(img_grid, bboxes2d)

        # source_features = torch.cat([roi_features, roi_grid], dim=1)
        roi_nocs = self.roi_layer(nocs, bboxes2d)
        # if seg_mask is not N
        roi_seg_mask = self.roi_layer(seg_mask, bboxes2d) if seg_mask is not None else None

        # denormalized nocs
        if normalize_nocs:
            # based on index,
            # generate the cooresponding dim, roty, location
            pred_dim = bboxes3d_cam[:, 4:7]
            pred_center = bboxes3d_cam[:, 1:4]
            pred_center[:,1] -= pred_dim[:,1]/2
            pred_yaw = bboxes3d_cam[:, 7:8]
            pred_dim = pred_dim.reshape(-1, 3, 1, 1)
            pred_dim = pred_dim.expand(-1, 3,  roi_nocs.shape[2], roi_nocs.shape[3])
            # pred_center =
            pred_yaw = pred_yaw.reshape(-1, 1, 1, 1)

            pred_yaw = pred_yaw.expand(-1, 1, roi_nocs.shape[2], roi_nocs.shape[3])
            # gravity center
            roi_nocs = self.nocs_coder.decode(roi_nocs, pred_dim, pred_yaw).detach()

        # sample candidate depth
        # check how to generate the intrinsic

        intrinsics = torch.stack(
            [features.new_tensor(i['cam2img'][0]) for i in img_metas], dim=0)
        extrinsics = torch.stack(
            [features.new_tensor(i['lidar2cam'][0]) for i in img_metas], dim=0)

        index = bboxes2d[:, 0].long()
        batch_intrinsics = torch.index_select(intrinsics, 0, index.long())
        batch_extrinsics = torch.index_select(extrinsics, 0, index.long())



        pred_uv = points_cam2img_batch(pred_center, batch_intrinsics)[..., :2].detach()

        pred_depth = pred_center[:, 2].unsqueeze(1) #K, 1
        pred_depth = pred_depth.expand(-1, self.depth_grid.shape[1])
        candidate_depth = pred_depth + self.depth_grid.expand(pred_depth.shape[0], -1)


        pred_uv = pred_uv.unsqueeze(1).expand(-1, self.depth_grid.shape[1], 2).reshape(-1, 2)

        batch_intrinsics_extend = batch_intrinsics.unsqueeze(1).expand(-1, self.depth_grid.shape[1], -1, -1)
        batch_intrinsics_extend = batch_intrinsics_extend.reshape(-1, 4, 4)
        candidate_center = points_img2cam_batch(
                        pred_uv, candidate_depth.reshape(-1, 1), batch_intrinsics_extend)

        candidate_center = candidate_center.reshape(len(pred_depth), -1, 3, 1, 1)
        candidate_center = candidate_center.expand(-1, -1, -1, roi_nocs.shape[2], roi_nocs.shape[3])

        roi_nocs = roi_nocs.unsqueeze(1).expand(-1, candidate_center.shape[1], -1, -1, -1)
        K, D, _, roi_H, roi_W = roi_nocs.shape

        roi_nocs = roi_nocs + candidate_center
        roi_nocs = roi_nocs.permute(0, 1, 3, 4, 2)
        temp_intrinsics = batch_intrinsics.permute(0, 2, 1)
        roi_nocs_2d = points_cam2img_broadcast(roi_nocs, temp_intrinsics.reshape(-1, 1, 1, 1, 4, 4))
        roi_nocs_2d = roi_nocs_2d[..., :2]


        roi_nocs_2d = roi_nocs_2d.reshape(K*D, roi_H, roi_W, 2)
        # roi_nocs_2d = roi_
        project_geo_features = roi_nocs_2d.clone()
        project_geo_features = project_geo_features.reshape(K, D, roi_H, roi_W, 2)
        project_geo_features = project_geo_features.permute(0, 1, 4, 2, 3)
        roi_nocs_2d = roi_nocs_2d / self.stride

        roi_nocs_2d[:,:,:,0]/= W
        roi_nocs_2d[:,:,:,1]/= H

        roi_nocs_2d = roi_nocs_2d * 2 - 1
        roi_nocs_2d = roi_nocs_2d.detach()
        batch_features = torch.index_select(features, 0, index.long())
        '''
        batch_features = batch_features.unsqueeze(1).expand(-1, D, -1, -1, -1)
        batch_features = batch_features.reshape(K*D, C, H, W)
        project_semantic_features = F.grid_sample(
                        batch_features, roi_nocs_2d,
                        mode="bilinear", align_corners=True, padding_mode="zeros")
        '''
        roi_nocs_2d = roi_nocs_2d.reshape(K, D, roi_H, roi_W, 2)
        project_semantic_features = [F.grid_sample(batch_features,
                                                   roi_nocs_2d[:,idx],
                                                   mode="bilinear",
                                                   align_corners=True,
                                                   padding_mode="zeros").unsqueeze(1) for idx in range(D)]
        project_semantic_features = torch.cat(project_semantic_features, dim=1).reshape(K, D, C, roi_H, roi_W)

        del batch_features

        # project_features = torch.cat([project_geo_features, project_semantic_features], dim=2)

        roi_features = roi_features.unsqueeze(1).expand(-1, D, -1, -1, -1)
        roi_grid = roi_grid.unsqueeze(1).expand(-1, D, -1, -1, -1)


        semantic_cost_volumes = torch.cat([roi_features, project_semantic_features], dim=2)
        geometric_cost_volumes = torch.cat([roi_grid, project_geo_features], dim=2)
        # FROM K D C+2, H, W to K C+2 D H W
        if roi_seg_mask is not None:
            roi_seg_mask = roi_seg_mask.unsqueeze(1).expand(-1, D, -1, -1, -1)
            semantic_cost_volumes *= \
                roi_seg_mask.expand(-1, -1, semantic_cost_volumes.shape[2], -1, -1) > self.foreground_threshold
            geometric_cost_volumes *= \
                    roi_seg_mask.expand(-1, -1, geometric_cost_volumes.shape[2], -1, -1) > self.foreground_threshold

        semantic_cost_volumes = semantic_cost_volumes.permute(0, 2, 1, 3, 4)
        geometric_cost_volumes = geometric_cost_volumes.permute(0, 2, 1, 3, 4)
        return (semantic_cost_volumes, geometric_cost_volumes)

def get2d_bboxes(bbox_list, img_metas, device="cuda", with_index=True, with_3d=False):
    bboxes2d_list = []
    if with_3d:
        bboxes3d_list = []
        bboxes3d_list_cam = []
    for idx in range(len(img_metas)):

        intrinsic = img_metas[idx]["cam2img"][0]
        extrinsic = img_metas[idx]["lidar2cam"][0]
        intrinsic = torch.tensor(intrinsic).to(device)
        extrinsic = torch.tensor(extrinsic).to(device)
        bboxes_cam = bbox_list[idx][0].convert_to(Box3DMode.CAM, extrinsic)
        bboxes_2d = projected_2d_box(bboxes_cam,
                    rt_mat=intrinsic, img_shape=img_metas[idx]['img_shape'][0],)
        if with_index:
            bboxes2d_list.append(torch.cat(
                [
                    bboxes_2d.new_ones(len(bboxes_2d), 1) * idx,
                    bboxes_2d], dim=1))
        else:
            bboxes2d_list.append(bboxes_2d)
        if with_3d:
            if with_index:
                bboxes3d_list.append(torch.cat(
                    [
                        bboxes_2d.new_ones(len(bboxes_2d), 1) * idx,
                        bbox_list[idx][0].tensor], dim=1))
                bboxes3d_list_cam.append(torch.cat(
                    [
                        bboxes_2d.new_ones(len(bboxes_2d), 1) * idx,
                        bboxes_cam.tensor], dim=1))
            else:
                bboxes3d_list.append(bbox_list[idx][0].tensor)
                bboxes3d_list_cam.append(bboxes_cam.tensor)

    bboxes2d_list = torch.cat(bboxes2d_list, dim=0)
    if with_3d:
        bboxes3d_list = torch.cat(bboxes3d_list, dim=0)
        bboxes3d_list_cam = torch.cat(bboxes3d_list_cam, dim=0)
        return bboxes2d_list, bboxes3d_list, bboxes3d_list_cam
    else:
        return bboxes2d_list


def get_batch_lidar2img(img_metas, index):
    intrinsics = torch.stack(
        [i['cam2img'][0] for i in img_metas], dim=0)
    extrinsics = torch.stack(
        [i['lidar2cam'][0] for i in img_metas], dim=0)

    batch_intrinsics = torch.index_select(intrinsics, 0, index.long())
    batch_extrinsics = torch.index_select(extrinsics, 0, index.long())
    return batch_intrinsics, batch_extrinsics


class RefineCostNetV1(nn.Module):
    '''Refinement module for the cost volume
    Args:
        input_planes
        feat_planes
    
    '''
    def __init__(self, input_planes, feat_planes):
        super().__init__()
        self.feature_extractor2d = nn.Identity()
        self.feature_extractor3d = nn.Sequential(
                BasicBlock3dV2(input_planes*2+4, input_planes*2+4),
                self._get_conv(input_planes*2+4, feat_planes))

        self.reg_head = self._mlp_module(feat_planes, 1)

    def _mlp_module(self, input_channels, out_channels):
        return nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels, out_channels),)

    def forward(self, features, depth_grid):
        '''
        Args:
            features: tuple
                semantic features with shape of N 2C D H W
                geometry features with shape of N 4 D H W
            output:
                features with shape of N C D
        '''
        semantic_features, geometry_features = features
        # prev conv 
        input_features = torch.cat([semantic_features, geometry_features], dim=1)
        # post conv 
        # consider do the downsampling;
        features = self.feature_extractor2d(input_features)
        features = self.feature_extractor3d(features)

        features = features.mean([3, 4])
        N, C, D = features.shape
        # features = features.permute([0, 2, 1]).reshape(-1, C)
        # return features
        features = features.permute(0, 2, 1).reshape(-1, C)
        reg_depth = self.reg_head(features)
        reg_depth = reg_depth.reshape(N, D)
        reg_depth = F.softmax(reg_depth, dim=1)
        reg_depth = reg_depth * depth_grid.expand(N, -1)
        reg_depth = reg_depth.mean(1, keepdim=True)

        return reg_depth


    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 1), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
