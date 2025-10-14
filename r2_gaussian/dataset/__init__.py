#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.dataset.cameras import PseudoCamera
from r2_gaussian.utils.camera_utils import cameraList_from_camInfos
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.pose_utils import generate_random_poses_pickle


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}  # 伪相机字典 - 参考X-Gaussian-depth实现

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
    
    def getPseudoCameras(self, scale=1.0):
        """获取伪相机列表 - 参考X-Gaussian-depth实现"""
        return self.pseudo_cameras.get(scale, [])
    
    def generate_multi_gaussian_cameras(self, num_additional_views=10, 
                                       radius_range=(0.8, 1.2), height_range=(-0.3, 0.3)):
        """
        生成多高斯训练的额外相机视角 - 参考X-Gaussian实现
        
        Args:
            num_additional_views: 额外视角数量
            radius_range: 半径范围
            height_range: 高度范围
        
        Returns:
            pseudo_cameras: List of PseudoCamera objects
        """
        pseudo_cameras = []
        
        # 使用训练相机生成随机姿态
        poses = generate_random_poses_pickle(self.train_cameras)
        
        # 获取第一个训练相机的参数作为参考
        if len(self.train_cameras) > 0:
            ref_camera = self.train_cameras[0]
            FoVx = ref_camera.FoVx
            FoVy = ref_camera.FoVy
            image_width = ref_camera.image_width
            image_height = ref_camera.image_height
        else:
            # 默认参数
            FoVx = FoVy = 0.5
            image_width = image_height = 256
        
        # 创建伪标签相机
        for R, T in poses:
            pseudo_cam = PseudoCamera(
                R=R, T=T,
                FoVx=FoVx, FoVy=FoVy,
                width=image_width, height=image_height,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0
            )
            pseudo_cameras.append(pseudo_cam)
        
        return pseudo_cameras
    
    def generate_pseudo_labels(self, gaussians, renderer, pseudo_cameras):
        """
        使用当前模型生成伪标签 - 参考X-Gaussian实现
        
        Args:
            gaussians: 当前高斯模型
            renderer: 渲染器
            pseudo_cameras: 伪标签相机列表
        
        Returns:
            pseudo_labels: List of pseudo label images
        """
        pseudo_labels = []
        
        for pseudo_cam in pseudo_cameras:
            pseudo_label = pseudo_cam.generate_pseudo_label(gaussians, renderer)
            pseudo_labels.append(pseudo_label)
        
        return pseudo_labels
