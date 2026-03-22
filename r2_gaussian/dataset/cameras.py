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
import sys
import torch
from torch import nn
import numpy as np

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        scanner_cfg,
        R,
        T,
        angle,
        mode,
        FoVx,
        FoVy,
        image,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        depth_image=None,  # 新增深度图像支持
        depth_bounds=None,  # 新增深度范围支持
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.angle = angle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.mode = mode
        self.image_name = image_name
        
        # 深度相关参数 - 参考X-Gaussian实现
        self.depth_image = depth_image
        self.depth_bounds = depth_bounds
        self.bounds = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)  # 默认边界

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            .transpose(0, 1)
            .cuda()
        )
        # 在CPU上计算矩阵乘法以避免CUBLAS错误
        world_view_cpu = self.world_view_transform.cpu()
        proj_matrix_cpu = self.projection_matrix.cpu()
        self.full_proj_transform = (
            torch.matmul(
                world_view_cpu.unsqueeze(0),
                proj_matrix_cpu.unsqueeze(0)
            )
        ).squeeze(0).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
    def get_depth_constraint(self):
        """获取深度约束信息 - 参考X-Gaussian实现"""
        if self.depth_image is not None:
            return self.depth_image, self.depth_bounds
        return None, None
    
    def has_depth_info(self):
        """检查是否有深度信息"""
        return self.depth_image is not None


class PseudoCamera:
    """Pseudo label camera for multi-view training - 参考X-Gaussian实现"""
    
    def __init__(self, R, T, FoVx, FoVy, width, height, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        """
        初始化伪标签相机
        
        Args:
            R: 旋转矩阵
            T: 平移向量
            FoVx: X方向视场角
            FoVy: Y方向视场角
            width: 图像宽度
            height: 图像高度
            trans: 变换向量
            scale: 缩放因子
        """
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.mode = 1  # 使用cone beam模式，与r2-gaussian兼容
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        
        # 在CPU上计算矩阵乘法以避免CUBLAS错误
        world_view_cpu = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
        # 使用r2-gaussian的getProjectionMatrix函数签名
        proj_matrix_cpu = getProjectionMatrix(
            fovX=self.FoVx,
            fovY=self.FoVy,
            mode=1,  # 使用cone beam模式
            scanner_cfg=None  # PseudoCamera不需要scanner_cfg
        ).transpose(0,1).cpu()
        
        self.world_view_transform = world_view_cpu.cuda()
        self.projection_matrix = proj_matrix_cpu.cuda()
        self.full_proj_transform = (
            torch.matmul(
                world_view_cpu.unsqueeze(0),
                proj_matrix_cpu.unsqueeze(0)
            )
        ).squeeze(0).cuda()
        
        # 计算相机中心
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
    def generate_pseudo_label(self, gaussians, render_func):
        """使用当前模型生成伪标签"""
        with torch.no_grad():
            # 调用渲染函数，传入相机、高斯模型和管道参数
            pseudo_image = render_func(self, gaussians)["render"]
        return pseudo_image
