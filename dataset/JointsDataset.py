# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random

from utils.transforms import get_affine_transform, affine_transform, get_scale
from utils.cameras import project_pose_cpu

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    def __init__(self, model_name, root_id, max_people, num_views, color_rgb, dataset_dir, cam_list, \
        ori_image_size, image_size, heatmap_size, \
        sigma, space_size, space_center, voxels_per_axis, individual_space_size, \
        is_train=True, transform=None):
        self.model_name = model_name
        self.root_id = root_id
        self.max_people = max_people
        self.num_views = num_views
        self.color_rgb = color_rgb
        self.dataset_dir = dataset_dir
        self.ori_image_size = ori_image_size
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.target_type = 'gaussian'

        # relates to camera calibration
        self.sigma = sigma
    
        self.space_size = np.array(space_size)
        self.space_center = np.array(space_center)
        self.voxels_per_axis = np.array(voxels_per_axis)
        self.individual_space_size = np.array(individual_space_size)

        self.input_heatmap_src = 'image'
        
        self.data_augmentation = False
        self.transform = transform
        self.resize_transform, self.center, self.scale = self._get_resize_transform()
        self.inv_resize_transform, _, _ = self._get_resize_transform(inv=1)
        self.cameras = None
        self.db = []
    
    def _get_resize_transform(self, inv=0):
        r = 0
        c = np.array([self.ori_image_size[0] / 2.0, self.ori_image_size[1] / 2.0])
        s = get_scale((self.ori_image_size[0], self.ori_image_size[1]), self.image_size)
        trans = get_affine_transform(c, s, r, self.image_size, inv=inv)
        return trans, c, s

    def _rebuild_db(self):
        for idx in range(len(self.db)):
            db_rec = self.db[idx]

            # dataset only for testing: no gt 3d pose
            if 'joints_3d' not in db_rec:
                meta = {
                    'seq': db_rec['seq'],
                    'all_image_path': db_rec['all_image_path'],
                }
                target = np.zeros((1, 1, 1), dtype=np.float32)
                target = torch.from_numpy(target)
                self.db[idx] = {
                    'target': target,
                    'meta': meta
                }
                continue

            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            nposes = len(joints_3d)
            assert nposes <= self.max_people, 'too many persons'

            joints_3d_u = np.zeros((self.max_people, self.num_joints, 3))
            joints_3d_vis_u = np.zeros((self.max_people, self.num_joints))
            for i in range(nposes):
                joints_3d_u[i] = joints_3d[i][:, 0:3]
                joints_3d_vis_u[i] = joints_3d_vis[i]

            if isinstance(self.root_id, int):
                roots_3d = joints_3d_u[:, self.root_id]
            elif isinstance(self.root_id, list):
                roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            
            target = self.generate_target(joints_3d, joints_3d_vis)
            meta = {
                'num_person': nposes,
                'joints_3d': joints_3d_u,
                'joints_3d_vis': joints_3d_vis_u,
                'roots_3d': roots_3d,
                'bbox': target['bbox'],
                'seq': db_rec['seq'],
            }
            
            if 'all_image_path' in db_rec.keys():
                meta['all_image_path'] = db_rec['all_image_path']
            
            self.db[idx] = {
                'target': target,
                'meta': meta
            }
        
            if 'pred_pose2d' in db_rec.keys():
                self.db[idx]['pred_pose2d'] = db_rec['pred_pose2d']
        
        return

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = self.db[idx]

        # read images as input
        if 'all_image_path' in db_rec['meta'].keys():
            all_image_path = db_rec['meta']['all_image_path']
            all_input = []
            all_img_path = []
            for image_path in all_image_path:
                input = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if self.color_rgb:
                    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                
                input = cv2.warpAffine(input,
                    self.resize_transform, (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR)

                if self.transform:
                    input = self.transform(input)

                all_input.append(input)
                all_img_path.append(image_path)
            if self.model_name == 'faster_voxelpose':
                all_input = torch.stack(all_input, dim=0)
        else:
            all_input = torch.zeros((1, 1, 1, 1))
            all_img_path = None

        # generate input heatmap
        if self.input_heatmap_src == 'image':
            input_heatmaps = torch.zeros((1, 1, 1))

        elif self.input_heatmap_src == 'pred':
            
            assert 'pred_pose2d' in db_rec.keys() and db_rec['pred_pose2d'] is not None, 'Dataset must provide pred_pose2d'
            
            all_preds = db_rec['pred_pose2d']
            input_heatmaps = []
            for preds in all_preds:
                for n in range(len(preds)):
                    for i in range(len(preds[n])):
                        preds[n][i, :2] = affine_transform(preds[n][i, :2], self.resize_transform)

                input_heatmap = torch.from_numpy(self.generate_input_heatmap(preds))
                input_heatmaps.append(input_heatmap)

            if self.model_name == 'faster_voxelpose':
                input_heatmaps = torch.stack(input_heatmaps, dim=0)
            
        elif self.input_heatmap_src == 'gt':
            assert 'joints_3d' in db_rec['meta'], 'Dataset must provide gt joints_3d'
            joints_3d = db_rec['meta']['joints_3d']
            joints_3d_vis = db_rec['meta']['joints_3d_vis']
            seq = db_rec['meta']['seq']
            nposes = len(joints_3d)
            input_heatmaps = []
            
            # obtain projects 2d gt poses
            for c in range(self.num_views):
                joints_2d = []
                joints_vis = []
                for n in range(nposes):
                    pose = project_pose_cpu(joints_3d[n], self.cameras[seq][c])

                    x_check = np.bitwise_and(pose[:, 0] >= 0,
                                                pose[:, 0] <= self.ori_image_size[0] - 1)
                    y_check = np.bitwise_and(pose[:, 1] >= 0,
                                                pose[:, 1] <= self.ori_image_size[1] - 1)
                    check = np.bitwise_and(x_check, y_check)
                    vis = joints_3d_vis[n] > 0
                    vis[np.logical_not(check)] = 0
                    
                    for i in range(len(pose)):
                        pose[i] = affine_transform(pose[i], self.resize_transform)
                        if (np.min(pose[i]) < 0 or pose[i, 0] >= self.image_size[0]
                            or pose[i, 1] >= self.image_size[1]):
                                vis[i] = 0

                    joints_2d.append(pose)
                    joints_vis.append(vis)
                
                input_heatmap = self.generate_input_heatmap(joints_2d, joints_vis=joints_vis)
                # input_heatmap, _ = self.generate_target_heatmap(joints_2d, joints_vis=joints_vis)
                input_heatmap = torch.from_numpy(input_heatmap)
                input_heatmaps.append(input_heatmap)
            
            if self.model_name == 'faster_voxelpose':
                input_heatmaps = torch.stack(input_heatmaps, dim=0)
        
        target = None
        if self.model_name == 'faster_voxelpose':
            target = db_rec["target"]
        meta = db_rec["meta"]

        target_heatmaps = target_weights = target_3d = None
        
        if self.model_name == 'voxelpose':
            assert 'joints_3d' in db_rec['meta'], 'Dataset must provide gt joints_3d'
            joints_3d = db_rec['meta']['joints_3d']
            joints_3d_vis = db_rec['meta']['joints_3d_vis']
            seq = db_rec['meta']['seq']
            nposes = len(joints_3d)
            target_heatmaps = []
            target_weights = []
            
            # obtain projects 2d gt poses
            for c in range(self.num_views):
                joints_2d = []
                joints_vis = []
                for n in range(nposes):
                    pose = project_pose_cpu(joints_3d[n], self.cameras[seq][c])

                    x_check = np.bitwise_and(pose[:, 0] >= 0,
                                                pose[:, 0] <= self.ori_image_size[0] - 1)
                    y_check = np.bitwise_and(pose[:, 1] >= 0,
                                                pose[:, 1] <= self.ori_image_size[1] - 1)
                    check = np.bitwise_and(x_check, y_check)
                    vis = joints_3d_vis[n] > 0
                    vis[np.logical_not(check)] = 0
                    
                    for i in range(len(pose)):
                        pose[i] = affine_transform(pose[i], self.resize_transform)
                        if (np.min(pose[i]) < 0 or pose[i, 0] >= self.image_size[0]
                            or pose[i, 1] >= self.image_size[1]):
                                vis[i] = 0
                    
                    joints_2d.append(pose)
                    joints_vis.append(vis)

                target_heatmap, target_weight = self.generate_target_heatmap(joints_2d, joints_vis, num_people=meta['num_person'])
                target_heatmap = torch.from_numpy(target_heatmap)
                target_weight = torch.from_numpy(target_weight)
                target_heatmaps.append(target_heatmap)
                target_weights.append(target_weight)
            # target_heatmaps = torch.stack(target_heatmaps, dim=0)
            # target_weights = torch.stack(target_weights, dim=0)
            
            # meta['joints_3d_vis'] = np.repeat(np.reshape(joints_3d_vis, (-1, 1)), 3, axis=1)

            target_3d = self.generate_3d_target(joints_3d, num_people=meta['num_person'])
            target_3d = torch.from_numpy(target_3d)

        elif self.model_name == 'mvp':
            assert 'joints_3d' in db_rec['meta'], 'Dataset must provide gt joints_3d'
            joints_3d = db_rec['meta']['joints_3d']
            joints_3d_vis = db_rec['meta']['joints_3d_vis']
            seq = db_rec['meta']['seq']
            nposes = len(joints_3d)
            joints_2d_np_list = []
            joints_vis_np_list = []
            cam_list = []
            cam_intri_list = []
            cam_r_list = []
            cam_focal_list = [] 
            cam_t_list = []
            cam_standard_t_list = []

            # obtain projects 2d gt poses
            joints_2d = []
            joints_vis = []
            for c in range(self.num_views):
                for n in range(nposes):
                    pose = project_pose_cpu(joints_3d[n], self.cameras[seq][c])

                    x_check = np.bitwise_and(pose[:, 0] >= 0,
                                                pose[:, 0] <= self.ori_image_size[0] - 1)
                    y_check = np.bitwise_and(pose[:, 1] >= 0,
                                                pose[:, 1] <= self.ori_image_size[1] - 1)
                    check = np.bitwise_and(x_check, y_check)
                    vis = joints_3d_vis[n] > 0
                    vis[np.logical_not(check)] = 0
                    
                    for i in range(len(pose)):
                        pose[i] = affine_transform(pose[i], self.resize_transform)
                        if (np.min(pose[i]) < 0 or pose[i, 0] >= self.image_size[0]
                            or pose[i, 1] >= self.image_size[1]):
                                vis[i] = 0

                    joints_2d.append(pose)
                    joints_vis.append(vis)

                joints_2d_np = np.zeros((self.max_people, self.num_joints, 2))
                joints_vis_np = np.zeros((self.max_people, self.num_joints))
                
                for i in range(nposes):
                    joints_2d_np[i] = joints_2d[i]
                    joints_vis_np[i] = joints_vis[i]

                joints_2d_np_list.append(joints_2d_np)
                joints_vis_np_list.append(joints_vis_np)
                
            for i in range(db_rec['meta']['num_person']):
                cam = self.cameras[db_rec['meta']['seq']][i]

                cam_intri = np.eye(3, 3)
                cam_intri[0, 0] = float(cam['fx'])
                cam_intri[1, 1] = float(cam['fy'])
                cam_intri[0, 2] = float(cam['cx'])
                cam_intri[1, 2] = float(cam['cy'])
                cam_R = cam['R']
                cam_T = cam['T']
                cam_standard_T = cam['standard_T']
                cam_focal = np.stack([cam['fx'], cam['fy'],
                                      np.ones_like(cam['fy'])])
                
                cam_list.append(cam)
                cam_intri_list.append(cam_intri)
                cam_r_list.append(cam_R)
                cam_focal_list.append(cam_focal)
                cam_t_list.append(cam_T)
                cam_standard_t_list.append(cam_standard_T)


            meta['joints'] = joints_2d_np_list
            meta['joints_vis'] = joints_vis_np_list
            meta['camera'] = cam_list
            meta['camera_Intri'] = cam_intri_list
            meta['camera_R'] = cam_r_list
            meta['camera_focal'] = cam_focal_list
            meta['camera_T'] = cam_t_list
            meta['camera_standard_T'] = cam_standard_t_list

        return all_img_path, all_input, target, meta, input_heatmaps, target_heatmaps, target_weights, target_3d

    def compute_human_scale(self, pose, joints_vis):
        idx = (joints_vis > 0.1)
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target_heatmap(self, joints, joints_vis, num_people):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = num_people #len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        # for i in range(num_joints):
        #     for n in range(nposes):
        #         if joints_vis[n][i, 0] == 1:
        #             target_weight[i, 0] = 1
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i]:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    # if joints_vis[n][joint_id, 0] == 0 or \
                    #         ul[0] >= self.heatmap_size[0] or \
                    #         ul[1] >= self.heatmap_size[1] \
                    #         or br[0] < 0 or br[1] < 0:
                    #     continue
                    if not joints_vis[n][joint_id] or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        # if self.use_different_joints_weight:
        #     target_weight = np.multiply(target_weight, self.joints_weight)
        return target, target_weight

    def generate_3d_target(self, joints_3d, num_people):
        # num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.voxels_per_axis
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target

    def generate_target(self, joints_3d, joints_3d_vis):
        # return: index, offset, bbox, 2d_heatmaps, 1d_heatmaps
        num_people = len(joints_3d)
        space_size = np.array(self.space_size)
        space_center = np.array(self.space_center)
        individual_space_size = np.array(self.individual_space_size)
        voxels_per_axis = np.array(self.voxels_per_axis)
        voxel_size = space_size / (voxels_per_axis - 1)

        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, voxels_per_axis[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, voxels_per_axis[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, voxels_per_axis[2]) + space_center[2]

        target_index = np.zeros((self.max_people))
        target_2d = np.zeros((voxels_per_axis[0], voxels_per_axis[1]), dtype=np.float32)
        target_1d = np.zeros((self.max_people, voxels_per_axis[2]), dtype=np.float32)
        target_bbox = np.zeros((self.max_people, 2), dtype=np.float32)
        target_offset = np.zeros((self.max_people, 2), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            idx = (joints_3d_vis[n] > 0.1)
            if isinstance(joint_id, int):
                center_pos = joints_3d[n][joint_id]
            elif isinstance(joint_id, list):
                center_pos = (joints_3d[n][joint_id[0]] + joints_3d[n][joint_id[1]]) / 2.0
            
            # compute target index, offset and bbox size
            loc = (center_pos - space_center + 0.5 * space_size) / voxel_size
            assert np.sum(loc < 0) == 0 and np.sum(loc > voxels_per_axis) == 0, "human centers out of bound!" 
            # flatten 2d index
            target_index[n] = (loc // 1)[0] * voxels_per_axis[1] + (loc // 1)[1]
            target_offset[n] = (loc % 1)[:2]
            target_bbox[n] = ((2 * np.abs(center_pos - joints_3d[n][idx]).max(axis = 0) + 200.0) / individual_space_size)[:2]
            if np.sum(target_bbox[n] > 1) > 0:
                print("Warning: detected an instance where the size of the bounding box is {:.2f}m, larger than 2m".format(np.max(target_bbox[n]) * 2.0))

            # Gaussian distribution
            mu_x, mu_y, mu_z = center_pos[0], center_pos[1], center_pos[2]
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            # generate 2d target
            gridx, gridy = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2) / (2 * cur_sigma ** 2))
            target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]] = np.maximum(target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]], g)

            # generate 1d target
            gridz = grid1Dz[i_z[0]:i_z[1]]
            g = np.exp(-(gridz - mu_z) ** 2 / (2 * cur_sigma ** 2))
            target_1d[n, i_z[0]:i_z[1]] = np.maximum(target_1d[n, i_z[0]:i_z[1]], g)
            
        target_2d = np.clip(target_2d, 0, 1)
        target_1d = np.clip(target_1d, 0, 1)
        mask = (np.arange(self.max_people) <= num_people)
        target = {'index': target_index, 'offset': target_offset, 'bbox': target_bbox,
                  '2d_heatmaps': target_2d, '1d_heatmaps': target_1d, 'mask':mask}
        return target

    def generate_input_heatmap(self, joints, joints_vis=None):
        num_joints = joints[0].shape[0]
        target = np.zeros((num_joints, self.heatmap_size[1],\
                           self.heatmap_size[0]), dtype=np.float32)
        feat_stride = self.image_size / self.heatmap_size

        for n in range(len(joints)):
            human_scale = 2 * self.compute_human_scale(
                    joints[n][:, :2] / feat_stride, np.ones(num_joints))
            if human_scale == 0:
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                if joints_vis is not None and joints_vis[n][joint_id] == 0:
                    continue

                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1]\
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                # if self.model_name == 'voxelpose' and 'shelf' in self.cfg.DATASET.TEST_DATASET:
                #     max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0
                #     # max_value = max_value**0.5
                # else:
                #     max_value = 1.0
                # g = np.exp(
                #     -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2)) * max_value
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                # data augmentation
                if self.data_augmentation:
                    # random scaling
                    scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                    if joint_id in [7, 8]:
                        scale = scale * 0.5 if random.random() < 0.1 else scale
                    elif joint_id in [9, 10]:
                        scale = scale * 0.2 if random.random() < 0.1 else scale
                    else:
                        scale = scale * 0.5 if random.random() < 0.05 else scale
                    g *= scale

                    # random occlusion
                    start = [int(np.random.uniform(0, self.heatmap_size[1] -1)),
                                int(np.random.uniform(0, self.heatmap_size[0] -1))]
                    end = [int(min(start[0] + np.random.uniform(self.heatmap_size[1] / 4, 
                            self.heatmap_size[1] * 0.75), self.heatmap_size[1])),
                            int(min(start[1] + np.random.uniform(self.heatmap_size[0] / 4,
                            self.heatmap_size[0] * 0.75), self.heatmap_size[0]))]
                    g[start[0]:end[0], start[1]:end[1]] = 0.0

                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1],
                                    img_x[0]:img_x[1]] = np.maximum(
                                        target[joint_id][img_y[0]:img_y[1],
                                                        img_x[0]:img_x[1]],
                                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            
            target = np.clip(target, 0, 1)

        return target
