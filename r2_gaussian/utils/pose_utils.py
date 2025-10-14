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

import torch
import numpy as np
import random
from math import pi, cos, sin


def look_at_matrix(camera_pos, target):
    """计算look-at矩阵 - 参考X-Gaussian-depth实现"""
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)

    up = np.array([0, 1, 0])
    if np.abs(np.dot(forward, up)) > 0.99:  # 如果接近平行，重新定义up
        up = np.array([1, 0, 0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # 归一化

    up = np.cross(forward, right)
    R = np.vstack([right, up, forward]).T
    return R


def generate_random_poses_360(num_poses=10, radius=1.0, height_range=(-0.5, 0.5)):
    """
    生成360度环绕的随机相机姿态 - 参考X-Gaussian实现
    
    Args:
        num_poses: 生成的姿态数量
        radius: 相机距离原点的半径
        height_range: 高度范围 (min_height, max_height)
    
    Returns:
        poses: List of (R, T) tuples
    """
    poses = []
    
    for i in range(num_poses):
        # 随机角度
        theta = random.uniform(0, 2 * pi)
        
        # 随机高度
        height = random.uniform(height_range[0], height_range[1])
        
        # 计算相机位置
        x = radius * cos(theta)
        y = height
        z = radius * sin(theta)
        
        # 相机朝向原点
        forward = np.array([-x, -y, -z])
        forward = forward / np.linalg.norm(forward)
        
        # 计算右向量
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        # 计算上向量
        up = np.cross(right, forward)
        
        # 构建旋转矩阵
        R = np.array([right, up, forward])
        
        # 平移向量
        T = np.array([x, y, z])
        
        poses.append((R, T))
    
    return poses


def generate_random_poses_llff(num_poses=10, radius_range=(0.8, 1.2), height_range=(-0.3, 0.3)):
    """
    生成LLFF风格的随机相机姿态 - 参考X-Gaussian实现
    
    Args:
        num_poses: 生成的姿态数量
        radius_range: 半径范围 (min_radius, max_radius)
        height_range: 高度范围 (min_height, max_height)
    
    Returns:
        poses: List of (R, T) tuples
    """
    poses = []
    
    for i in range(num_poses):
        # 随机半径
        radius = random.uniform(radius_range[0], radius_range[1])
        
        # 随机角度
        theta = random.uniform(0, 2 * pi)
        
        # 随机高度
        height = random.uniform(height_range[0], height_range[1])
        
        # 计算相机位置
        x = radius * cos(theta)
        y = height
        z = radius * sin(theta)
        
        # 相机朝向原点
        forward = np.array([-x, -y, -z])
        forward = forward / np.linalg.norm(forward)
        
        # 计算右向量
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        # 计算上向量
        up = np.cross(right, forward)
        
        # 构建旋转矩阵
        R = np.array([right, up, forward])
        
        # 平移向量
        T = np.array([x, y, z])
        
        poses.append((R, T))
    
    return poses


def generate_random_poses_pickle(views):
    """生成适合pickle数据的随机相机姿态 - 简化版本"""
    n_poses = 50  # 减少生成数量，避免计算过载
    poses = []
    
    # 获取场景中心点
    if len(views) > 0:
        # 计算所有相机位置的中心
        positions = []
        for view in views:
            positions.append(view.T)
        center = np.mean(positions, axis=0)
    else:
        center = np.array([0, 0, 0])
    
    # 生成随机姿态
    for _ in range(n_poses):
        # 随机半径和角度
        radius = np.random.uniform(0.5, 1.5)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        
        # 计算相机位置
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        
        camera_pos = np.array([x, y, z])
        
        # 计算朝向中心的旋转矩阵
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        if np.abs(np.dot(forward, up)) > 0.99:
            up = np.array([1, 0, 0])
        
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        up = np.cross(forward, right)
        
        R = np.array([right, up, forward]).T
        T = camera_pos
        
        poses.append((R, T))
    
    return poses


def poses_avg(poses):
    """计算姿态的平均值"""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
    """构造look-at视图矩阵"""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x):
    """归一化向量"""
    return x / np.linalg.norm(x)


def pad_poses(poses):
    """为姿态矩阵添加齐次坐标"""
    if poses.ndim == 2:  # 单个4x4矩阵
        return np.concatenate([poses, np.array([[0, 0, 0, 1]])], axis=0)
    else:  # 批量矩阵
        return np.concatenate([poses, np.tile(np.array([[[0, 0, 0, 1]]]), [poses.shape[0], 1, 1])], axis=1)


def unpad_poses(poses):
    """移除姿态矩阵的齐次坐标"""
    return poses[:, :3, :4]


def generate_uniform_poses_forview(view, n_frames=20, z_variation=0.1, z_phase=0):
    """生成均匀分布的相机姿态 - 参考X-Gaussian-depth实现"""
    # 根据views计算景物的中心点
    radius = 0.5
    camera_pos = view.T
    target_point = view.T + view.R[:, 2] * 5
    poses = []

    for i in range(n_frames):
        # 计算圆周上的均匀分布点
        angle = 2 * np.pi * i / n_frames
        x = radius * np.cos(angle) + camera_pos[0]
        y = radius * np.sin(angle) + camera_pos[1]
        z = camera_pos[2] + np.random.uniform(-z_variation, z_variation)  # 随机高度

        camera_pos_new = np.array([x, y, z])
        R = look_at_matrix(camera_pos_new, target_point)

        poses.append({'R': R, 'T': camera_pos_new})

    return poses


def interpolate_poses(pose1, pose2, alpha):
    """
    在两个相机姿态之间插值
    
    Args:
        pose1: 第一个姿态 (R1, T1)
        pose2: 第二个姿态 (R2, T2)
        alpha: 插值参数 (0-1)
    
    Returns:
        interpolated_pose: 插值后的姿态 (R, T)
    """
    R1, T1 = pose1
    R2, T2 = pose2
    
    # 线性插值平移
    T_interp = (1 - alpha) * T1 + alpha * T2
    
    # 球面线性插值旋转
    R_interp = slerp_rotation(R1, R2, alpha)
    
    return (R_interp, T_interp)


def slerp_rotation(R1, R2, alpha):
    """
    球面线性插值旋转矩阵
    
    Args:
        R1: 第一个旋转矩阵
        R2: 第二个旋转矩阵
        alpha: 插值参数 (0-1)
    
    Returns:
        R_interp: 插值后的旋转矩阵
    """
    # 转换为四元数进行插值
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    
    # 四元数球面线性插值
    q_interp = slerp_quaternion(q1, q2, alpha)
    
    # 转换回旋转矩阵
    R_interp = quaternion_to_rotation_matrix(q_interp)
    
    return R_interp


def rotation_matrix_to_quaternion(R):
    """旋转矩阵转四元数"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def slerp_quaternion(q1, q2, alpha):
    """四元数球面线性插值"""
    # 计算点积
    dot = np.dot(q1, q2)
    
    # 如果点积为负，取反其中一个四元数以选择最短路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 如果点积接近1，使用线性插值避免数值不稳定
    if dot > 0.9995:
        result = q1 + alpha * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 计算角度
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    
    theta = alpha * theta_0
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2
