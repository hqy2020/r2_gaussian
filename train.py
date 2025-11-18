###############################################################
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# 本软件仅限非商业、科研和评估用途，具体条款见 LICENSE.md 文件。
# 如有疑问请联系 george.drettakis@inria.fr
###############################################################

import os
import os.path as osp
import torch
import torch.nn.functional as F
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import matplotlib.pyplot as plt

# 添加项目路径，导入自定义模块

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams  # 参数定义
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian  # 高斯模型相关
from r2_gaussian.utils.general_utils import safe_state  # 随机种子等系统状态
from r2_gaussian.utils.cfg_utils import load_config  # 配置文件加载
from r2_gaussian.utils.log_utils import prepare_output_and_logger  # 日志与输出
from r2_gaussian.dataset import Scene  # 数据集场景
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, loss_photometric, l1_loss_mask, depth_loss, pseudo_label_loss, depth_loss_fn, compute_graph_laplacian_loss  # 损失函数
from r2_gaussian.utils.depth_utils import extract_depth_from_volume_ray_casting  # 深度提取函数
from r2_gaussian.utils.warp_utils import inverse_warp  # 逆变形函数 - IPSM实现
from r2_gaussian.utils.image_utils import metric_vol, metric_proj  # 评估指标
from r2_gaussian.utils.plot_utils import show_two_slice  # 可视化工具
from r2_gaussian.utils.sghmc_optimizer import create_sss_optimizer, HybridOptimizer  # SSS优化器

# FSGS伪标签改进模块 (可选，向下兼容)
try:
    from r2_gaussian.utils.pseudo_view_utils import FSGSPseudoViewGenerator, create_fsgs_pseudo_cameras
    from r2_gaussian.utils.depth_estimator import MonocularDepthEstimator, create_depth_estimator
    # from r2_gaussian.utils.fsgs_improved import FSGSImprovedGenerator, create_improved_fsgs_pseudo_cameras
    HAS_FSGS_MODULES = True
    print("✅ FSGS pseudo-label modules available")
except ImportError as e:
    HAS_FSGS_MODULES = False
    print(f"📦 FSGS modules not available: {e}")
    print("📦 Falling back to legacy pseudo-label implementation")

# Medical Proximity-guided密化模块 (新增)
try:
    from r2_gaussian.utils.realistic_proximity_guided import HighQualityMedicalProximityGuidedDensifier
    HAS_PROXIMITY_GUIDED = True
    print("✅ Medical Proximity-guided密化 modules available")
except ImportError as e:
    HAS_PROXIMITY_GUIDED = False
    print(f"📦 Proximity-guided modules not available: {e}")

# FSGS Proximity-guided密化模块 (性能优化版本 - 2025-11-15)
try:
    from r2_gaussian.utils.fsgs_proximity_optimized import (
        FSGSProximityDensifierOptimized as FSGSProximityDensifier,
        add_fsgs_proximity_to_gaussian_model_optimized as add_fsgs_proximity_to_gaussian_model
    )
    HAS_FSGS_PROXIMITY = True
    print("✅ FSGS Proximity-guided densification modules available (OPTIMIZED)")
except ImportError as e:
    HAS_FSGS_PROXIMITY = False
    print(f"📦 FSGS Proximity modules not available: {e}")

# 🌟🌟 FSGS 完整系统模块 (完��实现 - 2025-11-15)
try:
    from r2_gaussian.utils.fsgs_complete import create_fsgs_complete_system
    from r2_gaussian.utils.fsgs_depth_renderer import FSGSDepthRenderer
    HAS_FSGS_COMPLETE = True
    print("✅ FSGS Complete System available (Proximity + Depth Supervision + Pseudo Views)")
except ImportError as e:
    HAS_FSGS_COMPLETE = False
    print(f"📦 FSGS Complete System not available: {e}")

# CoR-GS Stage 3 - Pseudo-view Co-regularization 模块 (2025-11-17)
try:
    from r2_gaussian.utils.pseudo_view_coreg import (
        generate_pseudo_view_medical,
        generate_random_pseudo_cameras,
        compute_pseudo_coreg_loss_medical
    )
    HAS_PSEUDO_COREG = True
    print("✅ CoR-GS Stage 3 (Pseudo-view Co-regularization) modules available")
except ImportError as e:
    HAS_PSEUDO_COREG = False
    print(f"📦 CoR-GS Stage 3 modules not available: {e}")
    print("📦 Falling back to baseline training (no pseudo-view co-regularization)")


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    gaussiansN=2,
    coreg=True,
    coprune=True,
    coprune_threshold=5,
    args=None,
):
    """
    训练主循环，负责高斯模型的初始化、损失计算、反向传播、稠密化与剪枝、保存模型和断点，以及日志记录。
    """
    first_iter = 0

    # 初始化数据集场景
    scene = Scene(dataset, shuffle=False)

    # 读取扫描仪配置和体素参数
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    # 查询函数，用于体素采样
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # ❌ 禁用 SSS（不确定实现是否正确）
    # 初始化标准高斯模型
    use_student_t = False  # 强制禁用 SSS
    print("📦 [R²] Using standard Gaussian model (SSS disabled)")
    gaussians = GaussianModel(scale_bound, use_student_t=False)

    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)

    # ❌ 禁用 SSS optimizer
    sss_optimizer = None

    # ❌ 禁用 GR-Gaussian（不确定实现是否正确）
    gr_graph = None
    print("⚠️ [R²] Graph Regularization disabled (focus on FSGS)")
    
    # FSGS Proximity-guided密化器初始化 (最新版本)
    proximity_densifier = None
    # ✅ 修复：支持从命令行参数和 dataset 两个来源激活 FSGS
    enable_fsgs_proximity = (
        getattr(args, 'enable_fsgs_proximity', False) or
        (hasattr(dataset, 'enable_fsgs_proximity') and dataset.enable_fsgs_proximity)
    )

    if enable_fsgs_proximity and HAS_FSGS_PROXIMITY:
        # 配置FSGS proximity参数 - 针对foot 3视角优化
        proximity_threshold = dataset.proximity_threshold if hasattr(dataset, 'proximity_threshold') else 8.0
        enable_medical_constraints = dataset.enable_medical_constraints if hasattr(dataset, 'enable_medical_constraints') else True
        organ_type = dataset.proximity_organ_type if hasattr(dataset, 'proximity_organ_type') else 'foot'
        
        # 为主高斯模型添加FSGS proximity功能
        gaussians = add_fsgs_proximity_to_gaussian_model(
            gaussians, 
            proximity_threshold=proximity_threshold,
            enable_medical_constraints=enable_medical_constraints,
            organ_type=organ_type
        )
        print(f"🌟 [FSGS-Proximity] Enabled for {organ_type} with threshold={proximity_threshold}")
    elif enable_fsgs_proximity:
        print("⚠️ [FSGS-Proximity] Module not available, falling back to standard densification")
    
    # 保留旧版本Proximity-guided密化器兼容性
    if hasattr(args, 'enable_proximity_guided') and args.enable_proximity_guided and HAS_PROXIMITY_GUIDED:
        proximity_densifier = HighQualityMedicalProximityGuidedDensifier()
        organ_type = getattr(args, 'proximity_organ_type', 'foot')
        print(f"🔬 [Legacy Proximity-Guided] Enabling medical proximity-guided densification for {organ_type}")
    elif hasattr(args, 'enable_proximity_guided') and args.enable_proximity_guided:
        print("⚠️ [Legacy Proximity-Guided] Module not available, falling back to standard densification")
    
    # 创建高斯场字典 - 参考X-Gaussian-depth实现
    GsDict = {}
    for i in range(gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        else:
            GsDict[f"gs{i}"] = GaussianModel(scale_bound, use_student_t=use_student_t)
            initialize_gaussian(GsDict[f"gs{i}"], dataset, None)
            GsDict[f"gs{i}"].training_setup(opt)
            if use_student_t:
                print(f"🎓 [SSS-R²] Create gaussians{i} with Student's t distribution")
            else:
                print(f"📦 [R²] Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")
    
    # 🌟🌟 FSGS 完整系统初始化 (Proximity + Depth + Pseudo Views - 2025-11-15)
    fsgs_system = None
    # 🔧 FIX: 不依赖 dataset 属性,直接使用 args
    enable_fsgs_complete = (
        enable_fsgs_proximity and
        HAS_FSGS_COMPLETE and
        (getattr(args, 'enable_fsgs_proximity', False) or  # 有 proximity 就启用
         getattr(args, 'enable_fsgs_pseudo_views', False))  # 或有 pseudo views
    )

    if enable_fsgs_complete:
        print("\n" + "="*60)
        print("🎯 Initializing FSGS Complete System")
        print("="*60)

        try:
            # 创建 FSGS 完整系统
            fsgs_system = create_fsgs_complete_system(dataset)

            # 初始化伪相机（在训练相机加载后）
            train_cameras = scene.getTrainCameras()
            fsgs_system.initialize_pseudo_cameras(train_cameras)

            print("✅ FSGS Complete System initialized successfully!")
            print("   - Proximity Unpooling: ✅")
            print("   - Depth Supervision: ✅" if fsgs_system.enable_depth_supervision else "   - Depth Supervision: ❌")
            print("   - Pseudo Views: ✅" if fsgs_system.enable_pseudo_views else "   - Pseudo Views: ❌")
            print("="*60 + "\n")

        except Exception as e:
            print(f"⚠️  FSGS Complete System initialization failed: {e}")
            print("   Falling back to proximity-only mode")
            fsgs_system = None
            enable_fsgs_complete = False

    # FSGS伪标签功能初始化 (向下兼容，仅在未使用完整系统时)
    pseudo_cameras = None
    pseudo_labels = None
    depth_estimator = None
    enable_fsgs = False  # 初始化（FSGS Complete模式下不使用旧版深度监督）

    if not enable_fsgs_complete:
        fsgs_generator = None
        enable_fsgs = getattr(args, 'enable_fsgs_pseudo', False) if args else False

        if dataset.multi_gaussian or dataset.pseudo_labels:
            if enable_fsgs and HAS_FSGS_MODULES:
                # 选择FSGS版本: improved 或 original
                fsgs_version = getattr(args, 'fsgs_version', 'improved') if args else 'improved'

                # 暂时只使用原版FSGS，避免导入问题
                if fsgs_version == 'improved':
                    print("🎯 [FSGS-Original] Using original FSGS (improved temporarily disabled)...")
                    fsgs_version = 'original'

                if fsgs_version == 'original':
                    print("🎯 [FSGS-Original] Using original FSGS implementation...")

                    # 创建原版FSGS风格伪视角生成器
                    fsgs_generator = FSGSPseudoViewGenerator(
                        noise_std=getattr(args, 'fsgs_noise_std', 0.05) if args else 0.05
                    )

                    # 生成原版FSGS风格伪相机
                    pseudo_cameras = fsgs_generator.generate_pseudo_cameras(
                        scene.train_cameras,
                        num_views=dataset.num_additional_views,
                        device=gaussians._xyz.device
                    )

                    print(f"✅ [FSGS-Original] Generated {len(pseudo_cameras)} original FSGS pseudo cameras")

                # 初始化深度估计器 (如果需要)
                depth_model_type = getattr(args, 'fsgs_depth_model', 'dpt_large') if args else 'dpt_large'
                if depth_model_type != 'disabled':
                    depth_estimator = create_depth_estimator(
                        model_type=depth_model_type,
                        device=gaussians._xyz.device,
                        enable_fsgs_depth=True
                    )
                    print(f"✅ [FSGS] Depth estimator: {depth_model_type}")
                else:
                    depth_estimator = None
                    print("📦 [FSGS] Depth estimator disabled")

            else:
                print("📦 [Legacy] Using original pseudo-label implementation...")
                pseudo_cameras = scene.generate_multi_gaussian_cameras(
                    num_additional_views=dataset.num_additional_views
                )
                print(f"Generated {len(pseudo_cameras)} legacy pseudo cameras")
    # 加载断点（如有）
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint from {osp.basename(checkpoint)}.")

    # 设置损失函数（是否使用 TV 损失）
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")

    # 🎯 CoR-GS: 定义背景颜色 (默认黑色)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 🎯 CoR-GS: 预生成随机 pseudo-view 相机（官方策略）
    pseudo_cameras_corgs = None
    if (args is not None and hasattr(args, 'enable_pseudo_coreg') and args.enable_pseudo_coreg and
        HAS_PSEUDO_COREG and gaussiansN >= 2):
        print("\n" + "="*60)
        print("🎯 [CoR-GS] 生成 10,000 个随机 pseudo-view 相机...")
        print("="*60)
        train_cameras = scene.getTrainCameras()
        pseudo_cameras_corgs = generate_random_pseudo_cameras(
            train_cameras=train_cameras,
            num_pseudo=10000,
            radius_range=(0.8, 1.2),
            seed=42
        )
        print(f"✅ [CoR-GS] Pseudo-view 生成完成: {len(pseudo_cameras_corgs)} 个")
        print("="*60 + "\n")

    if use_tv:
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # 训练主循环
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # 更新学习率 - 为每个高斯场更新
        for i in range(gaussiansN):
            GsDict[f"gs{i}"].update_learning_rate(iteration)

        # 随机选择一个训练视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 为每个高斯场渲染 - 参考X-Gaussian-depth实现
        RenderDict = {}
        for i in range(gaussiansN):
            RenderDict[f"render_pkg_gs{i}"] = render(
                viewpoint_cam,
                GsDict[f'gs{i}'],
                pipe,
                enable_drop=args.enable_drop,
                drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                iteration=iteration,
            )
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]

        # 计算每个高斯场的损失
        LossDict = {}
        gt_image = viewpoint_cam.original_image.cuda()
        
        for i in range(gaussiansN):
            LossDict[f"loss_gs{i}"] = l1_loss(RenderDict[f"image_gs{i}"], gt_image)
            
            # DSSIM 损失
        if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(RenderDict[f"image_gs{i}"], gt_image)
                LossDict[f"loss_gs{i}"] += opt.lambda_dssim * loss_dssim
        
        # 协同训练 - 参考X-Gaussian-depth实现
        if coreg and gaussiansN > 1:
            for i in range(gaussiansN):
                for j in range(gaussiansN):
                    if i != j:
                        coreg_loss = l1_loss(RenderDict[f"image_gs{i}"], RenderDict[f"image_gs{j}"].detach())
                        LossDict[f"loss_gs{i}"] += coreg_loss
        
        # 多高斯训练损失 - 原始版本（identity loss）
        if dataset.multi_gaussian and pseudo_cameras is not None and gaussiansN > 1:
            for pseudo_cam in pseudo_cameras[:3]:  # 限制数量避免计算过载
                for i in range(gaussiansN):
                    pseudo_render_pkg = render(
                        pseudo_cam,
                        GsDict[f'gs{i}'],
                        pipe,
                        enable_drop=args.enable_drop,
                        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                        iteration=iteration,
                    )
                    pseudo_image = pseudo_render_pkg["render"]
                    # 原始错误版本：identity loss（自己和自己比较）
                    LossDict[f"loss_gs{i}"] += dataset.multi_gaussian_weight * l1_loss(pseudo_image, pseudo_image.detach())
        
        # FSGS伪标签训练损失 (可选，向下兼容)
        # FSGS延迟启动: 2000次迭代后启动 (原版: 1000次)
        fsgs_start_iter = 2000 if enable_fsgs_proximity else 1000
        if dataset.pseudo_labels and pseudo_cameras is not None and iteration > fsgs_start_iter:
            # 获取伪相机和对应的最近真实相机
            pseudo_stack, closest_cam_stack = scene.getPseudoCamerasWithClosestViews(pseudo_cameras)
            
            if len(pseudo_stack) > 0:
                # 创建副本（IPSM的做法）
                pseudo_stack = pseudo_stack.copy()
                closest_cam_stack = closest_cam_stack.copy()
                
                # 随机选择一个伪相机（IPSM的drop机制）
                randint_idx = randint(0, len(pseudo_stack) - 1)
                pseudo_cam = pseudo_stack.pop(randint_idx)
                closest_cam = closest_cam_stack.pop(randint_idx)
                
                for j in range(gaussiansN):
                    # 从伪相机渲染图像
                    pseudo_render_pkg = render(
                        pseudo_cam,
                        GsDict[f'gs{j}'],
                        pipe,
                    )
                    rendered_img_pseudo = pseudo_render_pkg["render"]  # (C, H, W)
                    H, W = rendered_img_pseudo.shape[1], rendered_img_pseudo.shape[2]
                    
                    # 从伪相机提取深度图（使用现有的depth提取方法）
                    tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                        bbox[1] - tv_vol_sVoxel - bbox[0]
                    ) * torch.rand(3)
                    vol_pred_pseudo = query(
                        GsDict[f"gs{j}"],
                        tv_vol_center,
                        tv_vol_nVoxel,
                        tv_vol_sVoxel,
                        pipe,
                    )["vol"]
                    rendered_depth_pseudo = extract_depth_from_volume_ray_casting(
                        vol_pred_pseudo,
                        pseudo_cam,
                        threshold=getattr(dataset, 'depth_threshold', 0.01)
                    )  # (H_vol, W_vol) - volume的尺寸，不是图像尺寸
                    
                    # 从最近真实相机获取图像和深度
                    closest_image_1 = closest_cam.original_image.cuda()  # (C, H_closest, W_closest)
                    closest_H, closest_W = closest_image_1.shape[1], closest_image_1.shape[2]
                    
                    # 从最近真实相机提取深度图
                    vol_pred_closest = query(
                        GsDict[f"gs{j}"],
                        tv_vol_center,
                        tv_vol_nVoxel,
                        tv_vol_sVoxel,
                        pipe,
                    )["vol"]
                    closest_depth_1 = extract_depth_from_volume_ray_casting(
                        vol_pred_closest,
                        closest_cam,
                        threshold=getattr(dataset, 'depth_threshold', 0.01)
                    )  # (H_vol, W_vol) - volume的尺寸，不是图像尺寸
                    
                    # 确保深度图尺寸与图像尺寸匹配（resize深度图到图像尺寸）
                    # 伪相机深度图resize到伪相机图像尺寸
                    pseudo_depth_H, pseudo_depth_W = rendered_depth_pseudo.shape
                    if pseudo_depth_H != H or pseudo_depth_W != W:
                        # 使用双线性插值将深度图resize到图像尺寸
                        rendered_depth_pseudo_resized = F.interpolate(
                            rendered_depth_pseudo.unsqueeze(0).unsqueeze(0),  # (1, 1, H_vol, W_vol)
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # (H, W)
                    else:
                        rendered_depth_pseudo_resized = rendered_depth_pseudo
                    
                    # 真实相机深度图resize到真实相机图像尺寸
                    closest_depth_H, closest_depth_W = closest_depth_1.shape
                    if closest_depth_H != closest_H or closest_depth_W != closest_W:
                        closest_depth_1_resized = F.interpolate(
                            closest_depth_1.unsqueeze(0).unsqueeze(0),  # (1, 1, H_vol, W_vol)
                            size=(closest_H, closest_W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # (H_closest, W_closest)
                    else:
                        closest_depth_1_resized = closest_depth_1
                    
                    # 构建内参矩阵（从FoV计算）- 使用伪相机的尺寸，因为target_depth是伪相机的
                    focal_x = pseudo_cam.image_width / (2.0 * np.tan(pseudo_cam.FoVx / 2.0))
                    focal_y = pseudo_cam.image_height / (2.0 * np.tan(pseudo_cam.FoVy / 2.0))
                    intrinsic = torch.tensor([
                        [focal_x, 0, pseudo_cam.image_width / 2.0],
                        [0, focal_y, pseudo_cam.image_height / 2.0],
                        [0, 0, 1]
                    ], device=closest_image_1.device, dtype=torch.float32)
                    
                    # 逆变形（inverse warp）- 使用resize后的深度图
                    # 注意：inverse_warp内部会使用source_image的尺寸，所以需要确保深度图尺寸匹配
                    warp_rst_1 = inverse_warp(
                        closest_image_1,
                        closest_depth_1_resized,  # 使用resize后的深度图
                        rendered_depth_pseudo_resized.unsqueeze(0),  # (1, H, W) - 使用resize后的深度图
                        closest_cam.world_view_transform,  # r2使用world_view_transform而不是extrinsic_matrix
                        pseudo_cam.world_view_transform,
                        intrinsic
                    )
                    
                    # 计算masked损失（完全按照IPSM图片代码）
                    # 注意：mask是Float类型，使用乘法代替位运算&
                    combined_mask = (warp_rst_1["mask_warp"] * warp_rst_1["mask_depth_strict"]).unsqueeze(0)  # (1, H, W)
                    
                    warped_masked_strict_image = warp_rst_1["warped_img"] * combined_mask
                    pseudo_masked_strict_image = rendered_img_pseudo * combined_mask
                    
                    # 损失缩放因子（逐渐增加，IPSM的做法）
                    loss_scale = min(iteration / 500.0, 1.0)
                    
                    # 计算masked L1损失
                    Ll1_masked_pseudo = l1_loss_mask(
                        pseudo_masked_strict_image,
                        warped_masked_strict_image.detach()
                    )
                    
                    # 添加到总损失（乘以loss_scale）
                    LossDict[f"loss_gs{j}"] += dataset.pseudo_label_weight * loss_scale * Ll1_masked_pseudo
                    
                    # 可选：每500次迭代打印一次信息
                    if iteration % 500 == 0:
                        mask_valid_ratio = combined_mask.sum().item() / (H * W)
                        print(f"[IPSM Drop] Iteration {iteration}, GS{j}: masked_loss={Ll1_masked_pseudo.item():.6f}, "
                              f"loss_scale={loss_scale:.3f}, valid_mask_ratio={mask_valid_ratio:.3f}")
        
        # FSGS深度监督 (伪视角+训练视角深度约束)
        if enable_fsgs and depth_estimator and depth_estimator.enabled and iteration > fsgs_start_iter:
            fsgs_depth_weight = getattr(args, 'fsgs_depth_weight', 0.05) if args else 0.05
            
            for j in range(gaussiansN):
                # 1. 训练视角深度监督
                try:
                    # 估计当前训练视角的深度
                    gt_image_for_depth = gt_image.unsqueeze(0)  # [1, C, H, W]
                    estimated_depth = depth_estimator.estimate_depth(gt_image_for_depth, normalize=True)
                    
                    if estimated_depth is not None:
                        # 渲染当前视角的深度图
                        rendered_depth = RenderDict.get(f"depth_gs{j}")
                        if rendered_depth is not None:
                            # 计算Pearson相关性深度损失
                            depth_loss_train = depth_estimator.compute_pearson_loss(
                                rendered_depth, estimated_depth.squeeze(0)
                            )
                            LossDict[f"loss_gs{j}"] += fsgs_depth_weight * depth_loss_train
                            
                            if iteration % 500 == 0:
                                print(f"[FSGS] Iteration {iteration}, GS{j}: train_depth_loss={depth_loss_train.item():.6f}")
                                
                except Exception as e:
                    if iteration % 1000 == 0:  # 减少错误日志频率
                        print(f"Warning: FSGS train depth loss failed: {e}")
                
                # 2. 伪视角深度监督 (如果有伪相机)
                if pseudo_cameras and len(pseudo_cameras) > 0:
                    try:
                        # 随机选择一个伪相机进行深度监督
                        pseudo_cam = pseudo_cameras[randint(0, len(pseudo_cameras) - 1)]
                        
                        # 渲染伪视角
                        pseudo_render_pkg = render(pseudo_cam, GsDict[f'gs{j}'], pipe)
                        pseudo_image = pseudo_render_pkg["render"]
                        pseudo_depth = pseudo_render_pkg.get("depth")
                        
                        if pseudo_depth is not None:
                            # 估计伪视角深度
                            pseudo_image_for_depth = pseudo_image.unsqueeze(0)
                            estimated_pseudo_depth = depth_estimator.estimate_depth(pseudo_image_for_depth, normalize=True)
                            
                            if estimated_pseudo_depth is not None:
                                # 计算伪视角深度损失
                                depth_loss_pseudo = depth_estimator.compute_pearson_loss(
                                    pseudo_depth, estimated_pseudo_depth.squeeze(0)
                                )
                                LossDict[f"loss_gs{j}"] += fsgs_depth_weight * depth_loss_pseudo
                                
                                if iteration % 500 == 0:
                                    print(f"[FSGS] Iteration {iteration}, GS{j}: pseudo_depth_loss={depth_loss_pseudo.item():.6f}")
                                    
                    except Exception as e:
                        if iteration % 1000 == 0:  # 减少错误日志频率
                            print(f"Warning: FSGS pseudo depth loss failed: {e}")

        # 🌟🌟 FSGS Complete 深度监督 (Proximity + Depth + Pseudo Views - 2025-11-15)
        if enable_fsgs_complete and fsgs_system is not None:
            try:
                # 为每个高斯场计算深度监督loss
                for i in range(gaussiansN):
                    depth_loss_dict = fsgs_system.compute_depth_loss(
                        viewpoint_cam,
                        GsDict[f'gs{i}'],
                        pipe,
                        background,
                        iteration
                    )

                    # 添加深度loss到总loss
                    if depth_loss_dict['depth_loss'].item() > 0:
                        LossDict[f"loss_gs{i}"] += depth_loss_dict['depth_loss']

                        # 每500轮打印一次
                        if iteration % 500 == 0:
                            print(f"[FSGS Complete] Iteration {iteration}, GS{i}:")
                            print(f"  train_depth_loss={depth_loss_dict['train_depth_loss'].item():.6f}")
                            print(f"  pseudo_depth_loss={depth_loss_dict['pseudo_depth_loss'].item():.6f}")
                            print(f"  total_depth_loss={depth_loss_dict['depth_loss'].item():.6f}")

            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"⚠️  [FSGS Complete] Depth loss failed: {e}")

        # Depth损失 - 使用voxelization提取深度
        if dataset.enable_depth and dataset.depth_loss_weight > 0:
            for i in range(gaussiansN):
                # 使用voxelization获取density volume
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                    bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)
                vol_pred = query(
                    GsDict[f"gs{i}"],
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    pipe,
                )["vol"]
                
                # 从volume提取深度图
                depth_map = extract_depth_from_volume_ray_casting(
                    vol_pred, 
                    viewpoint_cam, 
                    threshold=dataset.depth_threshold
                )
                
                # 如果有ground truth深度，计算深度损失
                if hasattr(viewpoint_cam, 'depth_image') and viewpoint_cam.depth_image is not None:
                    gt_depth = viewpoint_cam.depth_image.cuda()
                    depth_loss_val = depth_loss_fn(
                        depth_map, 
                        gt_depth, 
                        loss_type=dataset.depth_loss_type
                    )
                    LossDict[f"loss_gs{i}"] += dataset.depth_loss_weight * depth_loss_val
                    
                # 自监督深度约束：让深度平滑，提升重建质量
                if depth_map.shape[0] > 1 and depth_map.shape[1] > 1:
                    # 计算深度图相邻像素的差异（水平+垂直）
                    depth_diff_h = torch.abs(depth_map[1:, :] - depth_map[:-1, :])
                    depth_diff_w = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])
                    consistency_loss = (depth_diff_h.mean() + depth_diff_w.mean()) * 0.1
                    
                    # 添加到总损失中
                    LossDict[f"loss_gs{i}"] += dataset.depth_loss_weight * consistency_loss
                    
                    # 每500次迭代打印一次
                    if iteration % 500 == 0:
                        print(f"[深度约束] Iteration {iteration}: {consistency_loss.item():.6f}")
        
        # 🌟 [GR-Gaussian] 图更新与图拉普拉斯正则化
        if dataset.enable_graph_laplacian:
            # 更新图结构 (每 graph_update_interval 次迭代,从 iteration 100 开始)
            if gr_graph is not None and iteration > 0 and iteration % dataset.graph_update_interval == 0:
                with torch.no_grad():
                    xyz = gaussians.get_xyz.detach()
                    gr_graph.build_knn_graph(xyz)
                    gr_graph.compute_edge_weights(xyz)
                    if iteration % 500 == 0:
                        print(f"[GR-Gaussian] Rebuilt graph at iteration {iteration}: "
                              f"{gr_graph.num_nodes} nodes, {gr_graph.edge_index.shape[1]} edges")

            # 计算图拉普拉斯损失 - 添加延迟启动和频率限制
            if iteration > 5000 and iteration % 500 == 0:  # 延迟启动 + 每500次迭代计算一次
                for i in range(gaussiansN):
                    graph_laplacian_loss = compute_graph_laplacian_loss(
                        GsDict[f"gs{i}"],
                        graph=gr_graph,  # 传递预构建的图 (如果存在)
                        k=dataset.graph_k,
                        Lambda_lap=dataset.graph_lambda_lap
                    )
                    LossDict[f"loss_gs{i}"] += graph_laplacian_loss

                    # 日志记录
                    if iteration % 500 == 0:
                        tb_writer.add_scalar(f'GR-Gaussian/graph_laplacian_gs{i}',
                                           graph_laplacian_loss.item(), iteration)
                    if iteration % 1000 == 0:
                        print(f"[GR-Gaussian] Iteration {iteration}, GS{i}: "
                              f"graph_loss={graph_laplacian_loss.item():.6f}")

        # 3D TV 损失 - 为每个高斯场计算
        if use_tv:
            for i in range(gaussiansN):
                # 随机选取一个小体积中心
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)
                vol_pred = query(
                GsDict[f"gs{i}"],
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                )["vol"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                LossDict[f"loss_gs{i}"] += opt.lambda_tv * loss_tv

        # === CoR-GS Stage 3: Pseudo-view Co-regularization (2025-11-17) ===
        # 🔧 Bug 修复版本：修复 Bug 1/2/3/4
        if (args is not None and hasattr(args, 'enable_pseudo_coreg') and args.enable_pseudo_coreg and
            HAS_PSEUDO_COREG and iteration >= args.pseudo_start_iter and gaussiansN >= 2 and
            pseudo_cameras_corgs is not None):

            try:
                # 步骤 1: 从预生成的 pseudo-view 列表中随机采样（修复 Bug 1）
                import random
                pseudo_camera = random.choice(pseudo_cameras_corgs)

                # 步骤 2: 渲染粗模型和精细模型的 pseudo-view
                renders_pseudo = []
                for gid in range(min(2, gaussiansN)):  # 仅对前两个模型进行 co-regularization
                    render_pkg_pseudo = render(
                        pseudo_camera,
                        GsDict[f'gs{gid}'],
                        pipe,
                        scaling_modifier=1.0,
                        enable_drop=args.enable_drop if hasattr(args, 'enable_drop') else False,
                        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                        iteration=iteration
                    )
                    renders_pseudo.append(render_pkg_pseudo)

                # 步骤 3: 计算 Co-regularization 损失（粗模型 vs 精细模型）
                # 🔧 修复 Bug 2/3: 分别计算每个模型的 disagreement loss，使用 detach()
                if len(renders_pseudo) >= 2:
                    render_gs0 = renders_pseudo[0]["render"]
                    render_gs1 = renders_pseudo[1]["render"]

                    # gs0 的损失：render_gs0 参与梯度，render_gs1 detach
                    loss_pseudo_coreg_dict_gs0 = compute_pseudo_coreg_loss_medical(
                        render1=render_gs0,
                        render2=render_gs1.detach(),  # 修复 Bug 2: detach
                        lambda_dssim=0.2,
                        roi_weights=None
                    )

                    # gs1 的损失：render_gs1 参与梯度，render_gs0 detach
                    loss_pseudo_coreg_dict_gs1 = compute_pseudo_coreg_loss_medical(
                        render1=render_gs1,
                        render2=render_gs0.detach(),  # 修复 Bug 2: detach
                        lambda_dssim=0.2,
                        roi_weights=None
                    )

                    # 步骤 4: 叠加到总损失（修复 Bug 3: 分别计算，避免梯度加倍）
                    # 修复 Bug 4: 确保 disagreement loss 参与优化
                    LossDict['loss_gs0'] += args.lambda_pseudo * loss_pseudo_coreg_dict_gs0['loss']
                    LossDict['loss_gs1'] += args.lambda_pseudo * loss_pseudo_coreg_dict_gs1['loss']

                    # 步骤 5: TensorBoard 日志记录
                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            "train_loss_patches/pseudo_coreg_total",
                            loss_pseudo_coreg.item(),
                            iteration
                        )
                        tb_writer.add_scalar(
                            "train_loss_patches/pseudo_coreg_l1",
                            loss_pseudo_coreg_dict['l1'].item(),
                            iteration
                        )
                        tb_writer.add_scalar(
                            "train_loss_patches/pseudo_coreg_dssim",
                            loss_pseudo_coreg_dict['d_ssim'].item(),
                            iteration
                        )
                        tb_writer.add_scalar(
                            "train_loss_patches/pseudo_coreg_ssim",
                            loss_pseudo_coreg_dict['ssim'].item(),
                            iteration
                        )
                        tb_writer.add_scalar(
                            "train_loss_patches/pseudo_coreg_weighted",
                            (args.lambda_pseudo * loss_pseudo_coreg).item(),
                            iteration
                        )

                    # 步骤 6: 控制台日志（每 100 iterations 输出一次）
                    if iteration % 100 == 0:
                        print(f"  [Pseudo Co-reg] Loss: {loss_pseudo_coreg.item():.6f}, "
                              f"L1: {loss_pseudo_coreg_dict['l1'].item():.6f}, "
                              f"SSIM: {loss_pseudo_coreg_dict['ssim'].item():.4f}, "
                              f"Weighted: {(args.lambda_pseudo * loss_pseudo_coreg).item():.6f}")

            except Exception as e:
                # 异常处理：打印警告但不中断训练（向下兼容）
                if iteration % 100 == 0:
                    import traceback
                    import sys
                    print(f"⚠️  [Pseudo Co-reg] Failed at iter {iteration}: {e}")
                    print("  → Full traceback:")
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    print(tb_str, file=sys.stderr)
                    print("  → Continuing training without pseudo-view loss...")

        # 🎯 [SSS-R²] Add regularization losses for Student's t parameters
        for i in range(gaussiansN):
            if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
                opacity = GsDict[f"gs{i}"].get_opacity
                nu = GsDict[f"gs{i}"].get_nu

                # 🎯 [SSS-v5-OPTIMAL] 最优正则化 - 基于 v4 诊断结果的折中方案
                # v4 诊断结论：
                #   - iter 5000: 23.44 dB (峰值) - balance_loss 0.001 能达到高性能
                #   - iter 10000: 13.13 dB (崩溃) - balance_loss 0.001 太弱，无法长期稳定
                # v5 策略：3倍权重 + 适度目标，平衡���能与稳定性

                # Opacity balance loss - 3倍 v4 权重 (0.001 → 0.003)
                pos_count = (opacity > 0).float().mean()
                pos_target = 0.75  # v4: 0.70 → v5: 0.75 (允许 25% 负值，比 v4 严格)
                balance_loss = torch.abs(pos_count - pos_target)
                LossDict[f"loss_gs{i}"] += 0.003 * balance_loss  # v5: 0.003 (3倍 v4)

                # Nu diversity loss: 保持 ν 多样性 (不影响 opacity)
                nu_diversity_loss = -torch.std(nu) * 0.1
                nu_range_loss = torch.mean(torch.relu(nu - 8.0)) + torch.mean(torch.relu(2.0 - nu))
                LossDict[f"loss_gs{i}"] += 0.001 * (nu_diversity_loss + nu_range_loss)

        # 🎯 [SSS-v5-OPTIMAL] Debug logging - 观察优化后的 opacity 动态
        if hasattr(GsDict[f"gs0"], 'use_student_t') and GsDict[f"gs0"].use_student_t and iteration % 2000 == 0:
            opacity = GsDict[f"gs0"].get_opacity
            nu = GsDict[f"gs0"].get_nu
            pos_ratio = (opacity > 0).float().mean()
            neg_ratio = (opacity < 0).float().mean()
            nu_mean = nu.mean()
            nu_std = nu.std()

            pos_target = 0.75  # v5目标

            print(f"🎯 [SSS-v5-OPTIMAL] Iter {iteration}")
            print(f"   Opacity: [{opacity.min():.3f}, {opacity.max():.3f}]")
            print(f"   Balance: {pos_ratio*100:.1f}% pos / {neg_ratio*100:.1f}% neg (target: {pos_target*100:.0f}% pos)")
            print(f"   Nu: mean={nu_mean:.2f}, std={nu_std:.2f}, range=[{nu.min():.1f}, {nu.max():.1f}]")

            # 记录极端情况
            extreme_neg = (opacity < -0.5).float().mean()
            extreme_pos = (opacity > 0.9).float().mean()
            print(f"   Extremes: {extreme_pos*100:.1f}% >0.9, {extreme_neg*100:.1f}% <-0.5")
        
        # 反向传播 - 为每个高斯场
        for i in range(gaussiansN):
            LossDict[f"loss_gs{i}"].backward(retain_graph=(i < gaussiansN - 1))

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # 自适应控制：更新高斯半径和统计 - 为每个高斯场
            for i in range(gaussiansN):
                viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                radii = RenderDict[f"radii_gs{i}"]
                
                GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter]
                )
                GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # 高斯点稠密化与剪枝 - 为每个高斯场
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    # 🔬 Proximity-Guided Densification (医学感知密化) 
                    if (proximity_densifier is not None and 
                        hasattr(args, 'proximity_interval') and 
                        iteration % args.proximity_interval == 0):
                        
                        organ_type = getattr(args, 'proximity_organ_type', 'foot')
                        max_points = getattr(args, 'proximity_max_points', 500)
                        
                        for i in range(gaussiansN):
                            current_gaussians = GsDict[f"gs{i}"].get_xyz  # (N, 3)
                            current_opacity = GsDict[f"gs{i}"].get_opacity  # (N, 1)
                            
                            print(f"🔬 [Proximity-Guided] Iter {iteration}: 分析GS{i}的医学合理性...")
                            
                            # 执行医学感知的proximity密化
                            densify_result = proximity_densifier.proximity_guided_densify_realistic(
                                current_gaussians, current_opacity, organ_type, max_points
                            )
                            
                            if densify_result['densified_points'] > 0:
                                new_positions = densify_result['new_positions']  # (K, 3) 
                                new_opacities = densify_result['new_opacities']  # (K, 1)
                                
                                # 创建新高斯点的其他属性 (基于近邻插值)
                                device = current_gaussians.device
                                num_new = new_positions.shape[0]
                                
                                # 初始化其他属性
                                new_colors = torch.zeros(num_new, 3, device=device)  # RGB
                                new_rotations = torch.zeros(num_new, 4, device=device)  # 四元数
                                new_rotations[:, 0] = 1.0  # w分量设为1 (单位四元数)
                                new_scales = torch.ones(num_new, 3, device=device) * 0.01  # 小尺度
                                
                                # 添加新高斯点到模型
                                GsDict[f"gs{i}"].densification_postfix(
                                    new_positions, new_colors, new_rotations, new_scales, new_opacities
                                )
                                
                                print(f"✅ [Proximity-Guided] GS{i}: 新增 {num_new} 个医学合理的高斯点")
                    
                    # 标准密化和剪枝流程
                    for i in range(gaussiansN):
                        # SSS: Apply stricter point control for Student's t distributions
                        if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
                            # Reduce max points for SSS to prevent performance issues
                            max_points_sss = min(opt.max_num_gaussians, 50000)  # Cap at 50k for SSS
                            current_points = GsDict[f"gs{i}"].get_xyz.shape[0]
                            
                            # More aggressive pruning for SSS
                            if current_points > max_points_sss * 0.8:  # Start aggressive pruning at 80% 
                                sss_grad_threshold = opt.densify_grad_threshold * 1.5  # Harder to densify
                                sss_density_threshold = opt.density_min_threshold * 0.8  # Easier to prune
                            else:
                                sss_grad_threshold = opt.densify_grad_threshold
                                sss_density_threshold = opt.density_min_threshold
                            
                            print(f"🎓 [SSS-Control] Iter {iteration}: GS{i} has {current_points} points (max: {max_points_sss})")
                            
                            # 使用增强版密化函数 (FSGS proximity-guided)
                            if hasattr(GsDict[f"gs{i}"], 'enhanced_densify_and_prune'):
                                print(f"✅ [Densify] Iter {iteration}: GS{i} 使用 FSGS enhanced_densify_and_prune")
                                GsDict[f"gs{i}"].enhanced_densify_and_prune(
                                    sss_grad_threshold,
                                    sss_density_threshold,
                                    opt.max_screen_size,
                                    max_scale,
                                    max_points_sss,  # Use SSS-specific limit
                                    densify_scale_threshold,
                                    bbox,
                                    enable_proximity_densify=enable_fsgs_proximity,
                                )
                            else:
                                # 回退到标准密化
                                print(f"⚠️ [Densify] Iter {iteration}: GS{i} 回退到标准 densify_and_prune (无FSGS)")
                                GsDict[f"gs{i}"].densify_and_prune(
                                    sss_grad_threshold,
                                    sss_density_threshold,
                                    opt.max_screen_size,
                                    max_scale,
                                    max_points_sss,  # Use SSS-specific limit
                                    densify_scale_threshold,
                                    bbox,
                                )
                        else:
                            # Standard densification for non-SSS gaussians
                            # 使用增强版密化函数 (FSGS proximity-guided)
                            if hasattr(GsDict[f"gs{i}"], 'enhanced_densify_and_prune'):
                                print(f"✅ [Densify] Iter {iteration}: GS{i} 使用 FSGS enhanced_densify_and_prune (标准模式)")
                                GsDict[f"gs{i}"].enhanced_densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                                    enable_proximity_densify=enable_fsgs_proximity,
                                )
                            else:
                                # 回退到标准密化
                                print(f"⚠️ [Densify] Iter {iteration}: GS{i} 回退到标准 densify_and_prune (无FSGS)")
                                GsDict[f"gs{i}"].densify_and_prune(
                                    opt.densify_grad_threshold,
                                    opt.density_min_threshold,
                                    opt.max_screen_size,
                                    max_scale,
                                    opt.max_num_gaussians,
                                    densify_scale_threshold,
                                    bbox,
                                )
            
            # Density decay功能 - 在densification开始后对密度进行衰减
            if dataset.opacity_decay and iteration > opt.densify_from_iter:
                opt.densify_until_iter = opt.iterations
                for i in range(gaussiansN):
                    GsDict[f"gs{i}"].density_decay(factor=0.995)
            
            # 检查高斯场是否为空
            for i in range(gaussiansN):
                if GsDict[f"gs{i}"].get_density.shape[0] == 0:
                    raise ValueError(
                        f"No Gaussian left in gs{i}. Change adaptive control hyperparameters!"
                    )

            # 优化器更新 - 为每个高斯场
            if iteration < opt.iterations:
                for i in range(gaussiansN):
                    # 🎯 [SSS-R²] Apply gradient clipping for stability
                    if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
                        # 固定梯度裁剪阈值,简化训练流程
                        nu_clip_norm = 0.5
                        opacity_clip_norm = 1.0
                        xyz_clip_norm = 2.0

                        # Nu parameter gradient clipping
                        if hasattr(GsDict[f"gs{i}"], '_nu') and GsDict[f"gs{i}"]._nu.grad is not None:
                            torch.nn.utils.clip_grad_norm_(GsDict[f"gs{i}"]._nu, max_norm=nu_clip_norm)

                        # Opacity parameter gradient clipping
                        if hasattr(GsDict[f"gs{i}"], '_opacity') and GsDict[f"gs{i}"]._opacity.grad is not None:
                            torch.nn.utils.clip_grad_norm_(GsDict[f"gs{i}"]._opacity, max_norm=opacity_clip_norm)

                        # Position gradient clipping (standard for all models)
                        if GsDict[f"gs{i}"]._xyz.grad is not None:
                            torch.nn.utils.clip_grad_norm_(GsDict[f"gs{i}"]._xyz, max_norm=xyz_clip_norm)
                    
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none=True)

            # 保存高斯模型
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)
                
                # 保存额外的高斯场
                if gaussiansN > 1:
                    for i in range(1, gaussiansN):
                        pcd_path = osp.join(scene.model_path, f"point_cloud_gs{i}/iteration_{iteration}")
                        os.makedirs(pcd_path, exist_ok=True)
                        GsDict[f"gs{i}"].save_ply(osp.join(pcd_path, "point_cloud.ply"))

            # 保存断点
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                checkpoint_data = {}
                for i in range(gaussiansN):
                    checkpoint_data[f"gs{i}"] = GsDict[f"gs{i}"].capture()
                checkpoint_data["iteration"] = iteration
                torch.save(checkpoint_data, ckpt_save_path + "/chkpnt" + str(iteration) + ".pth")

            # 进度条显示
            if iteration % 10 == 0:
                # 计算总损失和总点数
                total_loss = sum(LossDict[f"loss_gs{i}"].item() for i in range(gaussiansN))
                total_points = sum(GsDict[f"gs{i}"].get_density.shape[0] for i in range(gaussiansN))
                
                progress_bar.set_postfix(
                    {
                        "loss": f"{total_loss:.1e}",
                        "pts": f"{total_points:2.1e}",
                        "gs": f"{gaussiansN}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志记录与评估
            metrics = {}
            for i in range(gaussiansN):
                metrics[f"loss_gs{i}"] = LossDict[f"loss_gs{i}"].item()
                for param_group in GsDict[f"gs{i}"].optimizer.param_groups:
                    metrics[f"lr_gs{i}_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, it=iteration: render(
                    x,
                    y,
                    pipe,
                    enable_drop=args.enable_drop,
                    drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                    iteration=it,
                ),
                queryfunc,
                gaussiansN,
                GsDict=GsDict,  # 🎯 CoR-GS: 传递 GsDict
                pipe=pipe,      # 🎯 CoR-GS: 传递 pipe 参数
                background=background,  # 🎯 CoR-GS: 传递背景颜色
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    gaussiansN=1,
    GsDict=None,  # 🎯 CoR-GS: 传递 GsDict 以访问多个模型
    pipe=None,    # 🎯 CoR-GS: Pipeline 参数 (用于渲染)
    background=None,  # 🎯 CoR-GS: 背景颜色
):
    """
    训练过程中的评估与日志记录，包括训练统计、2D渲染性能、3D重建性能等。
    """
    # 🎯 [DEBUG] 调试 CoR-GS 日志记录
    if iteration % 500 == 0:
        print(f"[DEBUG-REPORT] Iter {iteration}: gaussiansN={gaussiansN}, GsDict={GsDict is not None}, tb_writer={tb_writer is not None}", flush=True)

    # 记录训练统计信息
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

        # ❌ 禁用 CoR-GS Disagreement 日志记录（不确定实现是否正确）
        # 专注于 FSGS 功能验证
        enable_corgs_logging = False  # 强制禁用
        if iteration % 500 == 0:  # 调试输出
            print(f"[DEBUG-CORGS-1] Iter {iteration}: enable_corgs_logging={enable_corgs_logging}", flush=True)
        if enable_corgs_logging:
            # 仅在指定频率记录 (避免额外计算开销)
            log_freq = 500  # 默认频率
            if iteration % log_freq == 0:
                print(f"[DEBUG-CORGS-2] Iter {iteration}: Entering CoR-GS logging block", flush=True)
                try:
                    from r2_gaussian.utils.corgs_metrics import log_corgs_metrics
                    print(f"[DEBUG-CORGS-3] Import successful", flush=True)

                    # 获取前两个 Gaussian 模型
                    gaussians_1 = GsDict.get("gs0", scene.gaussians)
                    gaussians_2 = GsDict.get("gs1", None)
                    print(f"[DEBUG-CORGS-4] gs2={gaussians_2 is not None}, pipe={pipe is not None}", flush=True)

                    if gaussians_2 is not None and pipe is not None:
                        # 准备参数
                        threshold = 0.3  # 默认阈值
                        test_cameras = scene.getTestCameras()
                        print(f"[DEBUG-CORGS-5] test_cameras length={len(test_cameras)}", flush=True)
                        if len(test_cameras) > 0:
                            test_camera = test_cameras[0]  # 使用第一个测试相机
                            bg_color = background if background is not None else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

                            # 计算所有 Disagreement 指标
                            corgs_metrics = log_corgs_metrics(
                                gaussians_1,
                                gaussians_2,
                                test_camera,
                                pipe,
                                bg_color,
                                threshold=threshold
                            )

                            # 记录到 TensorBoard
                            for metric_name, metric_value in corgs_metrics.items():
                                tb_writer.add_scalar(f"corgs/{metric_name}", metric_value, iteration)

                            # 打印日志 (可选)
                            print(f"[CoR-GS Metrics @ Iter {iteration}] "
                                  f"Fitness={corgs_metrics['point_fitness']:.4f}, "
                                  f"RMSE={corgs_metrics['point_rmse']:.6f}, "
                                  f"PSNR_diff={corgs_metrics['render_psnr_diff']:.2f} dB")

                except ImportError as e:
                    print(f"⚠️ CoR-GS metrics module not available: {e}")
                except Exception as e:
                    print(f"⚠️ Error computing CoR-GS metrics: {e}")

    # 测试与评估
    if iteration in testing_iterations:
        # 2D渲染性能评估
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # 渲染所有视角
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    
                    # 保存单独的渲染图像（PNG格式）
                    if idx in show_idx:
                        # 创建可视化输出目录
                        vis_output_dir = osp.join(eval_save_path, "render_images")
                        os.makedirs(vis_output_dir, exist_ok=True)
                        
                        # 保存GT图像
                        gt_img_np = gt_image[0].detach().cpu().numpy()
                        gt_img_np = np.clip(gt_img_np, 0, 1) * 255
                        gt_save_path = osp.join(vis_output_dir, f"{viewpoint.image_name}_gt.png")
                        plt.imsave(gt_save_path, gt_img_np, cmap='viridis')
                        
                        # 保存渲染图像  
                        render_img_np = image[0].detach().cpu().numpy()
                        render_img_np = np.clip(render_img_np, 0, 1) * 255
                        render_save_path = osp.join(vis_output_dir, f"{viewpoint.image_name}_render.png")
                        plt.imsave(render_save_path, render_img_np, cmap='viridis')
                        
                        # 保存对比图（差异图）
                        diff_img = np.abs(gt_img_np - render_img_np)
                        diff_save_path = osp.join(vis_output_dir, f"{viewpoint.image_name}_diff.png")
                        plt.imsave(diff_save_path, diff_img, cmap='hot')
                        
                        print(f"💾 保存渲染图像: {viewpoint.image_name} 到 {vis_output_dir}")
                    
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # 3D重建性能评估
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.4f}, ssim3d {ssim_3d:.4f}, psnr2d {psnr_2d:.4f}, ssim2d {ssim_2d:.4f}"
        )

        # 记录其他指标
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 命令行入口，参数解析与训练启动
    # fmt: off
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters") 
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)  # 是否开启异常检测
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])  # 测试迭代
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])  # 保存迭代
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 断点保存迭代
    parser.add_argument("--start_checkpoint", type=str, default=None)  # 起始断点
    parser.add_argument("--config", type=str, default=None)  # 配置文件路径
    parser.add_argument("--enable_drop", action="store_true", default=False)  # 是否启用 drop 方法
    parser.add_argument("--drop_rate", type=float, default=0.10)  # drop 比例（0~1）
    
    # SSS: Student Splatting and Scooping 参数
    parser.add_argument("--enable_sss", action="store_true", default=False)  # 是否启用SSS
    parser.add_argument("--sghmc_friction", type=float, default=0.1)  # SGHMC摩擦系数
    parser.add_argument("--sghmc_burnin_steps", type=int, default=1000)  # SGHMC烧入步数
    parser.add_argument("--nu_lr_init", type=float, default=0.001)  # nu参数初始学习率
    parser.add_argument("--opacity_lr_init", type=float, default=0.01)  # opacity参数初始学习率
    
    # FSGS Proximity-Guided Densification 参数在arguments/__init__.py中已定义
    
    # 旧版本 Proximity-Guided Densification 参数 (兼容性保留)
    parser.add_argument("--enable_proximity_guided", action="store_true", default=False)  # 是否启用旧版proximity-guided密化
    parser.add_argument("--proximity_interval", type=int, default=1000)  # proximity密化间隔
    parser.add_argument("--proximity_max_points", type=int, default=500)  # 每次proximity密化最大点数

    # CoR-GS Stage 3 参数 (Pseudo-view Co-regularization - 2025-11-17)
    parser.add_argument("--enable_pseudo_coreg", action="store_true", default=False,
                        help="启用 CoR-GS Stage 3 Pseudo-view Co-regularization")
    parser.add_argument("--lambda_pseudo", type=float, default=1.0,
                        help="Pseudo-view co-regularization 损失权重")
    parser.add_argument("--pseudo_noise_std", type=float, default=0.02,
                        help="Pseudo-view 相机位置随机扰动标准差")
    parser.add_argument("--pseudo_start_iter", type=int, default=0,
                        help="开始应用 pseudo-view co-reg 的 iteration")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # 初始化系统状态（如随机种子）
    safe_state(args.quiet)

    # 加载配置文件（如有）
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # 设置日志与输出
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    # 是否开启异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 启动训练主循环
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.gaussiansN,
        args.coreg,
        args.coprune,
        args.coprune_threshold,
        args,
    )
    
    # 注意：在训练过程中使用 render 函数时，需要传递 enable_drop=args.enable_drop 参数

    # 训练结束
    print("Training complete.")
