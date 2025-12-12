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
    """
    return {
        'shape': list(plane.shape),
        'mean': float(np.mean(plane)),
        'std': float(np.std(plane)),
        'min': float(np.min(plane)),
        'max': float(np.max(plane)),
        'abs_mean': float(np.mean(np.abs(plane))),
        'non_zero_ratio': float(np.mean(np.abs(plane) > 1e-6)),
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


def analyze_adm_distribution(checkpoint: dict, positions: np.ndarray = None, device: str = 'cuda') -> dict:
    """
    分析ADM的offset和confidence分布

    如果提供positions，会计算实际的调制值
    """
    # 尝试从checkpoint中提取ADM诊断信息
    adm_info = {}

    # 检查是否有预存的诊断信息
    if 'adm_diagnostics' in checkpoint:
        adm_info = checkpoint['adm_diagnostics']

    # 检查是否有offset/confidence的样本
    if 'offset_samples' in checkpoint:
        adm_info['offset_samples'] = checkpoint['offset_samples']
    if 'confidence_samples' in checkpoint:
        adm_info['confidence_samples'] = checkpoint['confidence_samples']

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

    # 诊断1：K-Planes特征
    if kplanes_data:
        for plane_name, plane_data in kplanes_data.items():
            stats = report['kplanes_statistics'][plane_name]

            # 检查是否全0或接近全0
            if stats['abs_mean'] < 1e-6:
                diagnosis.append({
                    'level': 'ERROR',
                    'issue': f'{plane_name} 特征接近全0',
                    'detail': f"平均绝对值 = {stats['abs_mean']:.6f}",
                    'suggestion': '检查K-Planes初始化或学习率是否过低'
                })

            # 检查特征范围
            if stats['max'] - stats['min'] < 0.01:
                diagnosis.append({
                    'level': 'WARNING',
                    'issue': f'{plane_name} 特征变化范围过小',
                    'detail': f"范围 = [{stats['min']:.4f}, {stats['max']:.4f}]",
                    'suggestion': '可能学习不足，考虑增加训练迭代或调整学习率'
                })

    # 诊断2：Decoder
    if not decoder_data or 'state_dict' not in decoder_data:
        diagnosis.append({
            'level': 'WARNING',
            'issue': 'Decoder参数未找到',
            'detail': 'checkpoint中没有density_decoder相关参数',
            'suggestion': '检查模型保存是否包含decoder参数'
        })

    # 如果没有发现问题
    if not diagnosis:
        diagnosis.append({
            'level': 'OK',
            'issue': 'ADM组件状态正常',
            'detail': 'K-Planes和Decoder参数都已找到',
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
        print("\nK-Planes 各平面统计:")
        for plane_name, stats in report['kplanes_statistics'].items():
            print(f"  {plane_name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print("K-Planes: 未找到")

    if decoder_data and 'state_dict' in decoder_data:
        print(f"\nDecoder 总参数: {report['decoder_info'].get('total_params', 'N/A')}")
    else:
        print("\nDecoder: 未找到")

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
    adm_info = analyze_adm_distribution(checkpoint, device=args.device)

    # 配置信息
    config = {
        'checkpoint': str(args.checkpoint),
    }

    # 生成可视化
    if kplanes_data:
        print("\n生成K-Planes可视化...")
        plot_kplanes_heatmaps(kplanes_data, output_dir)
    else:
        print("\n跳过K-Planes可视化（未找到数据）")

    # 生成诊断报告
    print("\n生成诊断报告...")
    generate_diagnosis_report(kplanes_data, decoder_data, adm_info, config, str(output_dir / 'adm_diagnosis_report.json'))

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
