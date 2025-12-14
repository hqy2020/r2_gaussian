#!/usr/bin/env python3
"""
ADM (Adaptive Density Modulation) 诊断工具

功能：
1. 从checkpoint加载K-Planes参数和Decoder
2. 可视化三个平面的特征图
3. 分析offset/confidence分布
4. 输出诊断报告

使用方法：
    python diagnose_adm.py --checkpoint output/xxx/ckpt/chkpnt30000.pth --output_dir diagnosis/adm/
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_checkpoint(checkpoint_path: str) -> dict:
    """
    加载模型checkpoint

    返回:
        dict: 包含 kplanes_encoder, density_decoder, 和其他模型状态
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == '.pth':
        data = torch.load(checkpoint_path, map_location='cpu')
    elif checkpoint_path.suffix == '.pickle':
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {checkpoint_path.suffix}")

    return data


def extract_kplanes_features(checkpoint: dict) -> dict:
    """
    从checkpoint提取K-Planes特征

    返回:
        dict: 包含 plane_xy, plane_xz, plane_yz 的numpy数组
    """
    kplanes_data = {}

    # 检查不同的键名
    encoder_keys = ['kplanes_encoder', 'kplanes_state', 'kplanes']

    for key in encoder_keys:
        if key in checkpoint:
            encoder_state = checkpoint[key]
            break
    else:
        # 如果没有找到encoder，尝试从整个state_dict中提取
        encoder_state = {}
        for k, v in checkpoint.items():
            if 'plane_' in k or 'kplanes' in k:
                encoder_state[k] = v

    if not encoder_state:
        return None

    # 提取三个平面
    for plane_name in ['plane_xy', 'plane_xz', 'plane_yz']:
        for k, v in encoder_state.items():
            if plane_name in k:
                if isinstance(v, torch.Tensor):
                    kplanes_data[plane_name] = v.cpu().numpy()
                else:
                    kplanes_data[plane_name] = np.array(v)
                break

    return kplanes_data if kplanes_data else None


def extract_decoder_params(checkpoint: dict) -> dict:
    """
    从checkpoint提取Decoder参数

    返回:
        dict: 包含 decoder 的权重和偏置
    """
    decoder_data = {}

    # 检查不同的键名
    decoder_keys = ['density_decoder', 'decoder_state', 'decoder']

    for key in decoder_keys:
        if key in checkpoint:
            decoder_state = checkpoint[key]
            decoder_data['state_dict'] = decoder_state
            break
    else:
        # 从整个state_dict中提取
        for k, v in checkpoint.items():
            if 'decoder' in k.lower() or 'offset_head' in k or 'confidence_head' in k:
                if 'state_dict' not in decoder_data:
                    decoder_data['state_dict'] = {}
                decoder_data['state_dict'][k] = v

    return decoder_data if decoder_data else None


def compute_plane_statistics(plane: np.ndarray) -> dict:
    """
    计算单个平面的统计量

    新增指标：
    - cv (变异系数): std/mean，衡量相对变化程度
    - spatial_std: 空间维度的标准差（衡量空间变化）
    - channel_variance: 不同通道间的方差
    """
    # 基础统计
    mean_val = float(np.mean(plane))
    std_val = float(np.std(plane))

    # 变异系数 (CV)
    cv = std_val / (abs(mean_val) + 1e-8)

    # 空间变化分析 (针对 [1, C, H, W] 或 [C, H, W] 格式)
    if plane.ndim == 4:
        plane_3d = plane[0]  # [C, H, W]
    else:
        plane_3d = plane  # [C, H, W]

    # 每个通道的空间标准差
    spatial_stds = [float(np.std(plane_3d[c])) for c in range(plane_3d.shape[0])]
    avg_spatial_std = float(np.mean(spatial_stds))

    # 通道间方差
    channel_means = [float(np.mean(plane_3d[c])) for c in range(plane_3d.shape[0])]
    channel_variance = float(np.var(channel_means))

    return {
        'shape': list(plane.shape),
        'mean': mean_val,
        'std': std_val,
        'min': float(np.min(plane)),
        'max': float(np.max(plane)),
        'abs_mean': float(np.mean(np.abs(plane))),
        'non_zero_ratio': float(np.mean(np.abs(plane) > 1e-6)),
        # 新增指标
        'cv': cv,  # 变异系数
        'avg_spatial_std': avg_spatial_std,  # 平均空间标准差
        'channel_variance': channel_variance,  # 通道间方差
    }


def plot_kplanes_heatmaps(kplanes_data: dict, output_dir: Path):
    """
    可视化K-Planes的三个平面

    为每个平面生成多通道特征的热力图
    """
    for plane_name, plane_data in kplanes_data.items():
        # plane_data shape: [1, C, H, W] 或 [C, H, W]
        if plane_data.ndim == 4:
            plane_data = plane_data[0]  # [C, H, W]

        C, H, W = plane_data.shape

        # 创建图表
        # 显示前8个通道（或所有通道如果少于8）
        n_channels = min(8, C)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_channels):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            channel_data = plane_data[i]
            im = ax.imshow(channel_data, cmap='RdBu', aspect='auto')
            ax.set_title(f'Channel {i}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 隐藏多余的subplot
        for i in range(n_channels, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f'K-Planes: {plane_name} (前{n_channels}个通道)', fontsize=14)
        plt.tight_layout()

        output_path = output_dir / f'kplanes_{plane_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 保存 {plane_name} 热力图: {output_path}")

        # 生成通道均值热力图
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        mean_data = np.mean(plane_data, axis=0)  # [H, W]
        im = ax.imshow(mean_data, cmap='viridis', aspect='auto')
        ax.set_title(f'K-Planes: {plane_name} (通道均值)', fontsize=14)
        ax.set_xlabel('X' if 'x' in plane_name else 'Y')
        ax.set_ylabel('Y' if 'y' in plane_name else 'Z')
        plt.colorbar(im, ax=ax, label='特征值')
        plt.tight_layout()

        output_path = output_dir / f'kplanes_{plane_name}_mean.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 保存 {plane_name} 均值热力图: {output_path}")


def plot_adm_distributions(adm_info: dict, output_dir: Path):
    """
    可视化 ADM offset 和 confidence 分布

    生成：
    1. offset 分布直方图
    2. confidence 分布直方图
    3. offset vs confidence 散点图
    4. effective_offset 分布直方图
    """
    if adm_info.get('status') != 'OK':
        print(f"⚠ 跳过 ADM 分布可视化: {adm_info.get('status', '未知状态')}")
        return

    samples = adm_info.get('samples', {})
    offset_samples = samples.get('offset', [])
    confidence_samples = samples.get('confidence', [])
    effective_offset_samples = samples.get('effective_offset', [])
    modulation_samples = samples.get('modulation', [])

    params = adm_info.get('params', {})
    max_range = float(params.get('max_range', 0.3))
    strength = float(params.get('strength', 1.0))
    view_scale = float(params.get('view_scale', 1.0))
    zero_mean = bool(params.get('zero_mean', False))

    if not offset_samples or not confidence_samples:
        print("⚠ 跳过 ADM 分布可视化: 没有样本数据")
        return

    offset_arr = np.array(offset_samples)
    confidence_arr = np.array(confidence_samples)
    effective_offset_arr = np.array(effective_offset_samples) if effective_offset_samples else None
    modulation_arr = np.array(modulation_samples) if modulation_samples else None

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Offset 分布
    ax1 = axes[0, 0]
    ax1.hist(offset_arr, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(x=offset_arr.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {offset_arr.mean():.4f}')
    ax1.set_xlabel('Offset [-1, 1]')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Offset Distribution (std={offset_arr.std():.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Confidence 分布
    ax2 = axes[0, 1]
    ax2.hist(confidence_arr, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Init (0.5)')
    ax2.axvline(x=confidence_arr.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {confidence_arr.mean():.4f}')
    ax2.set_xlabel('Confidence [0, 1]')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Confidence Distribution (std={confidence_arr.std():.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Offset vs Confidence 散点图
    ax3 = axes[1, 0]
    # 随机采样以避免过多点
    n_plot = min(5000, len(offset_arr))
    indices = np.random.choice(len(offset_arr), n_plot, replace=False)
    scatter = ax3.scatter(offset_arr[indices], confidence_arr[indices],
                         c=offset_arr[indices] * confidence_arr[indices],
                         cmap='RdBu', alpha=0.5, s=10)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Offset')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Offset vs Confidence (color=offset*conf)')
    plt.colorbar(scatter, ax=ax3)
    ax3.grid(True, alpha=0.3)

    # 4. Effective Offset 分布
    ax4 = axes[1, 1]
    if effective_offset_arr is None or effective_offset_arr.size == 0:
        effective_offset = offset_arr * confidence_arr * max_range
    else:
        effective_offset = effective_offset_arr

    ax4.hist(effective_offset, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax4.axvline(x=effective_offset.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {effective_offset.mean():.4f}')
    ax4.set_xlabel(
        f'Effective Offset (max_range={max_range}, strength={strength}, view_scale={view_scale}, zero_mean={zero_mean})'
    )
    ax4.set_ylabel('Count')
    ax4.set_title(f'Effective Offset Distribution (std={effective_offset.std():.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('ADM Distribution Analysis', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'adm_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存 ADM 分布图: {output_path}")


def analyze_adm_distribution(
    checkpoint: dict,
    positions: np.ndarray = None,
    device: str = 'cuda',
    max_range: float = 0.3,
    strength: float = 1.0,
    num_views: int = None,
    view_adaptive: bool = False,
    zero_mean: bool = False,
) -> dict:
    """
    分析ADM的offset和confidence分布

    通过加载 K-Planes encoder 和 decoder，计算实际的 offset/confidence 分布。

    参数:
        checkpoint: 加载的 checkpoint 数据
        positions: 高斯位置（如果为 None，从 checkpoint 提取）
        device: 计算设备
        max_range: ADM 最大调制范围（与训练侧 adm_max_range 对齐）
        strength: ADM 调度强度（与训练侧 strength 对齐；诊断时可手动指定）
        num_views: 训练视角数（用于模拟 view-adaptive 缩放）
        view_adaptive: 是否启用 view-adaptive 缩放（需要配合 num_views）
        zero_mean: 是否启用零均值调制（与训练侧 adm_zero_mean 对齐）

    返回:
        dict: 包含 offset, confidence, effective_offset, modulation 的统计信息
    """
    adm_info = {}

    # 检查是否有 K-Planes 和 Decoder 状态
    kplanes_state = checkpoint.get('kplanes_state')
    decoder_state = checkpoint.get('decoder_state')

    if kplanes_state is None or decoder_state is None:
        adm_info['status'] = 'K-Planes 或 Decoder 状态未找到'
        return adm_info

    # 获取高斯位置
    if positions is None:
        xyz = checkpoint.get('xyz')
        if xyz is None:
            adm_info['status'] = '高斯位置未找到'
            return adm_info
        positions = xyz

    try:
        # 导入必要模块
        from r2_gaussian.gaussian.kplanes import KPlanesEncoder, DensityMLPDecoder

        # 从 state_dict 推断参数
        # plane_xy shape: [1, feature_dim, resolution, resolution]
        plane_xy = kplanes_state.get('plane_xy')
        if plane_xy is None:
            adm_info['status'] = 'K-Planes plane_xy 未找到'
            return adm_info

        feature_dim = plane_xy.shape[1]
        resolution = plane_xy.shape[2]

        # 创建并加载 K-Planes encoder
        encoder = KPlanesEncoder(
            grid_resolution=resolution,
            feature_dim=feature_dim,
            num_levels=1,
            bounds=(-1.0, 1.0)
        ).to(device)
        encoder.load_state_dict(kplanes_state)
        encoder.eval()

        # 从 decoder state_dict 推断参数
        # backbone.0.weight shape: [hidden_dim, input_dim]
        backbone_weight = decoder_state.get('backbone.0.weight')
        if backbone_weight is None:
            adm_info['status'] = 'Decoder backbone 未找到'
            return adm_info

        hidden_dim = backbone_weight.shape[0]
        input_dim = backbone_weight.shape[1]

        # 计算层数 (backbone.0, backbone.2, backbone.4, ...)
        num_layers = 2  # 默认
        for key in decoder_state.keys():
            if key.startswith('backbone.') and key.endswith('.weight'):
                layer_idx = int(key.split('.')[1])
                num_layers = max(num_layers, layer_idx // 2 + 2)

        # 创建并加载 Decoder
        decoder = DensityMLPDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        decoder.load_state_dict(decoder_state)
        decoder.eval()

        # 转换位置到 tensor
        if isinstance(positions, np.ndarray):
            xyz_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        else:
            xyz_tensor = positions.to(device)

        # 计算 K-Planes 特征和 offset/confidence
        with torch.no_grad():
            kplanes_feat = encoder(xyz_tensor)  # [N, feature_dim * 3]
            offset, confidence = decoder(kplanes_feat)  # [N, 1], [N, 1]

            # 计算有效调制（尽量对齐训练侧实现）
            view_scale = 1.0
            if view_adaptive and num_views is not None and num_views > 0:
                import math
                view_scale = 1.0 / math.sqrt(num_views / 3.0)
                view_scale = max(view_scale, 0.3)

            max_range = float(max_range)
            strength = float(strength)
            effective_offset = offset * confidence * max_range * strength * view_scale

            if zero_mean:
                effective_offset = effective_offset - effective_offset.mean()
                max_abs = max_range * strength * view_scale
                if max_abs > 0:
                    effective_offset = torch.clamp(effective_offset, -max_abs, max_abs)
            modulation = 1.0 + effective_offset

            # 统计信息
            adm_info['offset'] = {
                'mean': float(offset.mean().item()),
                'std': float(offset.std().item()),
                'min': float(offset.min().item()),
                'max': float(offset.max().item()),
                'abs_mean': float(offset.abs().mean().item()),
                'percentile_5': float(torch.quantile(offset, 0.05).item()),
                'percentile_95': float(torch.quantile(offset, 0.95).item()),
            }
            adm_info['confidence'] = {
                'mean': float(confidence.mean().item()),
                'std': float(confidence.std().item()),
                'min': float(confidence.min().item()),
                'max': float(confidence.max().item()),
                'percentile_5': float(torch.quantile(confidence, 0.05).item()),
                'percentile_95': float(torch.quantile(confidence, 0.95).item()),
            }
            adm_info['effective_offset'] = {
                'mean': float(effective_offset.mean().item()),
                'std': float(effective_offset.std().item()),
                'min': float(effective_offset.min().item()),
                'max': float(effective_offset.max().item()),
                'abs_mean': float(effective_offset.abs().mean().item()),
            }
            adm_info['modulation'] = {
                'mean': float(modulation.mean().item()),
                'std': float(modulation.std().item()),
                'min': float(modulation.min().item()),
                'max': float(modulation.max().item()),
            }
            adm_info['num_gaussians'] = int(xyz_tensor.shape[0])
            adm_info['params'] = {
                'max_range': float(max_range),
                'strength': float(strength),
                'view_scale': float(view_scale),
                'num_views': int(num_views) if num_views is not None else None,
                'view_adaptive': bool(view_adaptive),
                'zero_mean': bool(zero_mean),
            }
            adm_info['status'] = 'OK'

            # 保存样本用于可视化（随机采样最多 10000 个点）
            n_samples = min(10000, xyz_tensor.shape[0])
            indices = torch.randperm(xyz_tensor.shape[0])[:n_samples]
            adm_info['samples'] = {
                'offset': offset[indices].cpu().numpy().flatten().tolist(),
                'confidence': confidence[indices].cpu().numpy().flatten().tolist(),
                'effective_offset': effective_offset[indices].cpu().numpy().flatten().tolist(),
                'modulation': modulation[indices].cpu().numpy().flatten().tolist(),
            }

    except Exception as e:
        adm_info['status'] = f'分析失败: {str(e)}'
        import traceback
        adm_info['traceback'] = traceback.format_exc()

    return adm_info


def generate_diagnosis_report(kplanes_data: dict, decoder_data: dict, adm_info: dict, config: dict, output_path: str):
    """
    生成ADM诊断报告
    """
    report = {
        'summary': {},
        'kplanes_statistics': {},
        'decoder_info': {},
        'adm_analysis': adm_info,
        'config': config,
        'diagnosis': []
    }

    # K-Planes统计
    if kplanes_data:
        for plane_name, plane_data in kplanes_data.items():
            report['kplanes_statistics'][plane_name] = compute_plane_statistics(plane_data)

        # 汇总
        total_params = sum(np.prod(p.shape) for p in kplanes_data.values())
        report['summary']['kplanes_total_params'] = int(total_params)
        report['summary']['kplanes_planes'] = list(kplanes_data.keys())
    else:
        report['summary']['kplanes_status'] = 'NOT_FOUND'

    # Decoder信息
    if decoder_data and 'state_dict' in decoder_data:
        decoder_params = sum(np.prod(v.shape) if hasattr(v, 'shape') else 0
                            for v in decoder_data['state_dict'].values()
                            if hasattr(v, 'shape'))
        report['decoder_info']['total_params'] = int(decoder_params)
        report['decoder_info']['layers'] = list(decoder_data['state_dict'].keys())
    else:
        report['decoder_info']['status'] = 'NOT_FOUND'

    # 自动诊断
    diagnosis = report['diagnosis']

    # 诊断阈值配置（可调整）
    STD_THRESHOLD = 0.01  # std 低于此值说明学习不足
    CV_THRESHOLD = 0.05   # 变异系数低于此值说明相对变化不足
    SPATIAL_STD_THRESHOLD = 0.005  # 空间标准差阈值

    # 诊断1：K-Planes特征
    if kplanes_data:
        all_stds = []
        all_cvs = []

        for plane_name, plane_data in kplanes_data.items():
            stats = report['kplanes_statistics'][plane_name]
            all_stds.append(stats['std'])
            all_cvs.append(stats['cv'])

            # 检查1: 是否全0或接近全0
            if stats['abs_mean'] < 1e-6:
                diagnosis.append({
                    'level': 'ERROR',
                    'issue': f'{plane_name} 特征接近全0',
                    'detail': f"平均绝对值 = {stats['abs_mean']:.6f}",
                    'suggestion': '检查K-Planes初始化或学习率是否过低'
                })

            # 检查2: 特征范围过小
            if stats['max'] - stats['min'] < 0.01:
                diagnosis.append({
                    'level': 'WARNING',
                    'issue': f'{plane_name} 特征变化范围过小',
                    'detail': f"范围 = [{stats['min']:.4f}, {stats['max']:.4f}]",
                    'suggestion': '可能学习不足，考虑增加训练迭代或调整学习率'
                })

            # 检查3: 🆕 std 过低（核心问题检测）
            if stats['std'] < STD_THRESHOLD:
                diagnosis.append({
                    'level': 'WARNING',
                    'issue': f'{plane_name} 标准差过低 (std={stats["std"]:.4f} < {STD_THRESHOLD})',
                    'detail': f"K-Planes 几乎没有学到空间变化，mean={stats['mean']:.4f} 接近初始化值 0.3",
                    'suggestion': '降低 TV 正则化权重 (lambda_plane_tv: 0.002 → 0.0005) 或增加学习率'
                })

            # 检查4: 🆕 变异系数过低
            if stats['cv'] < CV_THRESHOLD:
                diagnosis.append({
                    'level': 'INFO',
                    'issue': f'{plane_name} 变异系数过低 (CV={stats["cv"]:.4f} < {CV_THRESHOLD})',
                    'detail': f"特征相对于均值的变化不足",
                    'suggestion': '考虑增加训练迭代或调整学习率'
                })

            # 检查5: 🆕 空间标准差过低
            if stats['avg_spatial_std'] < SPATIAL_STD_THRESHOLD:
                diagnosis.append({
                    'level': 'WARNING',
                    'issue': f'{plane_name} 空间标准差过低 (avg_spatial_std={stats["avg_spatial_std"]:.4f})',
                    'detail': f"每个通道内的空间变化不足，TV 正则化可能过强",
                    'suggestion': '降低 lambda_plane_tv 或使用自适应 TV 调度'
                })

        # 汇总诊断：整体学习状态
        avg_std = np.mean(all_stds)
        avg_cv = np.mean(all_cvs)
        report['summary']['avg_std'] = float(avg_std)
        report['summary']['avg_cv'] = float(avg_cv)

        if avg_std < STD_THRESHOLD:
            diagnosis.append({
                'level': 'ERROR',
                'issue': f'K-Planes 整体学习不足 (avg_std={avg_std:.4f})',
                'detail': '所有平面的特征变化都很小，ADM 无法有效调制密度',
                'suggestion': '1) 降低 lambda_plane_tv: 0.002 → 0.0005\n'
                             '         2) 增加 kplanes_lr_init: 0.002 → 0.005\n'
                             '         3) 缩短 warmup: adm_warmup_iters 3000 → 1000'
            })

    # 诊断2：Decoder
    if not decoder_data or 'state_dict' not in decoder_data:
        diagnosis.append({
            'level': 'WARNING',
            'issue': 'Decoder参数未找到',
            'detail': 'checkpoint中没有density_decoder相关参数',
            'suggestion': '检查模型保存是否包含decoder参数'
        })

    # 诊断3：ADM offset/confidence 分布分析
    if adm_info.get('status') == 'OK':
        offset_stats = adm_info.get('offset', {})
        confidence_stats = adm_info.get('confidence', {})
        effective_stats = adm_info.get('effective_offset', {})

        # 检查 offset 标准差（应该有足够变化）
        if offset_stats.get('std', 0) < 0.1:
            diagnosis.append({
                'level': 'WARNING',
                'issue': f'Offset 变化不足 (std={offset_stats.get("std", 0):.4f} < 0.1)',
                'detail': f'Offset 分布过于集中，无法产生有效的密度调制',
                'suggestion': '检查 K-Planes 特征学习是否充分'
            })

        # 检查 confidence 是否偏向极端
        conf_mean = confidence_stats.get('mean', 0.5)
        if conf_mean < 0.3:
            diagnosis.append({
                'level': 'INFO',
                'issue': f'Confidence 偏低 (mean={conf_mean:.4f})',
                'detail': '网络倾向于抑制调制，可能是学习不足或初始化问题',
                'suggestion': '检查 decoder 初始化和学习率'
            })
        elif conf_mean > 0.7:
            diagnosis.append({
                'level': 'INFO',
                'issue': f'Confidence 偏高 (mean={conf_mean:.4f})',
                'detail': '网络倾向于强调制，可能过拟合',
                'suggestion': '考虑增加正则化或降低 adm_max_range'
            })

        # 检查有效调制范围
        eff_abs_mean = effective_stats.get('abs_mean', 0)
        if eff_abs_mean < 0.01:
            diagnosis.append({
                'level': 'WARNING',
                'issue': f'有效调制过小 (abs_mean={eff_abs_mean:.4f} < 0.01)',
                'detail': 'ADM 几乎没有产生实际的密度变化',
                'suggestion': '增加 adm_max_range 或检查 K-Planes 学习'
            })

        # 将 ADM 分析添加到 summary
        report['summary']['adm_analysis'] = {
            'offset_std': offset_stats.get('std', 0),
            'confidence_mean': conf_mean,
            'effective_abs_mean': eff_abs_mean,
            'num_gaussians': adm_info.get('num_gaussians', 0),
        }

    # 如果没有发现问题
    if not diagnosis:
        diagnosis.append({
            'level': 'OK',
            'issue': 'ADM组件状态正常',
            'detail': 'K-Planes和Decoder参数都已找到，特征学习充分',
            'suggestion': '无需调整'
        })

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"✓ 保存诊断报告: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("ADM 诊断报告摘要")
    print("="*60)

    if kplanes_data:
        print(f"K-Planes 总参数: {report['summary'].get('kplanes_total_params', 'N/A')}")
        print(f"平均 std: {report['summary'].get('avg_std', 'N/A'):.4f} (健康阈值 > 0.01)")
        print(f"平均 CV:  {report['summary'].get('avg_cv', 'N/A'):.4f} (健康阈值 > 0.05)")
        print("\nK-Planes 各平面统计:")
        for plane_name, stats in report['kplanes_statistics'].items():
            status = "✓" if stats['std'] >= STD_THRESHOLD else "⚠"
            print(f"  {status} {plane_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, cv={stats['cv']:.4f}, spatial_std={stats['avg_spatial_std']:.4f}")
    else:
        print("K-Planes: 未找到")

    if decoder_data and 'state_dict' in decoder_data:
        print(f"\nDecoder 总参数: {report['decoder_info'].get('total_params', 'N/A')}")
    else:
        print("\nDecoder: 未找到")

    # ADM 分析摘要
    adm_summary = report['summary'].get('adm_analysis')
    if adm_summary:
        print(f"\nADM 调制分析 ({adm_summary.get('num_gaussians', 'N/A')} 高斯点):")
        offset_std = adm_summary.get('offset_std', 0)
        conf_mean = adm_summary.get('confidence_mean', 0)
        eff_mean = adm_summary.get('effective_abs_mean', 0)
        status_offset = "✓" if offset_std >= 0.1 else "⚠"
        status_conf = "✓" if 0.3 <= conf_mean <= 0.7 else "⚠"
        status_eff = "✓" if eff_mean >= 0.01 else "⚠"
        print(f"  {status_offset} offset std: {offset_std:.4f} (健康阈值 > 0.1)")
        print(f"  {status_conf} confidence mean: {conf_mean:.4f} (健康范围 0.3-0.7)")
        print(f"  {status_eff} effective |offset| mean: {eff_mean:.4f} (健康阈值 > 0.01)")
    elif adm_info.get('status'):
        print(f"\nADM 分析: {adm_info.get('status')}")

    print("\n诊断结果:")
    for d in diagnosis:
        print(f"  [{d['level']}] {d['issue']}")
        print(f"         {d['detail']}")
        if d['suggestion']:
            print(f"         建议: {d['suggestion']}")
    print("="*60)

    return report


def main():
    parser = argparse.ArgumentParser(description='ADM 诊断工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint路径 (.pth 或 .pickle)')
    parser.add_argument('--output_dir', type=str, default='diagnosis/adm',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')
    parser.add_argument('--adm_max_range', type=float, default=0.3,
                        help='ADM 最大调制范围 (默认: 0.3)')
    parser.add_argument('--adm_strength', type=float, default=1.0,
                        help='ADM 调度强度（诊断侧缩放，默认: 1.0）')
    parser.add_argument('--num_views', type=int, default=None,
                        help='训练视角数（用于模拟 view-adaptive 缩放）')
    parser.add_argument('--adm_view_adaptive', action='store_true',
                        help='启用 view-adaptive 缩放（需要配合 --num_views）')
    parser.add_argument('--adm_zero_mean', action='store_true',
                        help='启用零均值调制（与训练侧 adm_zero_mean 对齐）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ADM 诊断工具")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  输出目录: {output_dir}")
    print()

    # 加载checkpoint
    print("加载checkpoint...")
    checkpoint = load_model_checkpoint(args.checkpoint)
    print(f"  加载完成，键: {list(checkpoint.keys())[:10]}...")

    # 提取K-Planes
    print("\n提取K-Planes特征...")
    kplanes_data = extract_kplanes_features(checkpoint)
    if kplanes_data:
        print(f"  找到 {len(kplanes_data)} 个平面")
        for name, data in kplanes_data.items():
            print(f"    {name}: shape={data.shape}")
    else:
        print("  未找到K-Planes数据")

    # 提取Decoder
    print("\n提取Decoder参数...")
    decoder_data = extract_decoder_params(checkpoint)
    if decoder_data:
        print(f"  找到Decoder参数")
    else:
        print("  未找到Decoder参数")

    # 分析ADM分布
    print("\n分析ADM分布...")
    adm_info = analyze_adm_distribution(
        checkpoint,
        device=args.device,
        max_range=args.adm_max_range,
        strength=args.adm_strength,
        num_views=args.num_views,
        view_adaptive=args.adm_view_adaptive,
        zero_mean=args.adm_zero_mean,
    )

    # 配置信息
    config = {
        'checkpoint': str(args.checkpoint),
        'adm_max_range': float(args.adm_max_range),
        'adm_strength': float(args.adm_strength),
        'num_views': int(args.num_views) if args.num_views is not None else None,
        'adm_view_adaptive': bool(args.adm_view_adaptive),
        'adm_zero_mean': bool(args.adm_zero_mean),
    }

    # 生成可视化
    if kplanes_data:
        print("\n生成K-Planes可视化...")
        plot_kplanes_heatmaps(kplanes_data, output_dir)
    else:
        print("\n跳过K-Planes可视化（未找到数据）")

    # 生成 ADM 分布可视化
    print("\n生成ADM分布可视化...")
    plot_adm_distributions(adm_info, output_dir)

    # 生成诊断报告
    print("\n生成诊断报告...")
    generate_diagnosis_report(kplanes_data, decoder_data, adm_info, config, str(output_dir / 'adm_diagnosis_report.json'))

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
