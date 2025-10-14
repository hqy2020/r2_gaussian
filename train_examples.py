#!/usr/bin/env python3
"""
R2-Gaussian增强版使用示例
参考X-Gaussian的多高斯、伪标签、深度功能实现

使用方法：
1. 基础训练（不变）
python train.py -s data/369/chest_50_3views.pickle -m output/basic

2. 多高斯训练
python train.py -s data/369/chest_50_3views.pickle -m output/multi --multi_gaussian

3. 伪标签训练
python train.py -s data/369/chest_50_3views.pickle -m output/pseudo --pseudo_labels

4. 深度约束训练
python train.py -s data/369/chest_50_3views.pickle -m output/depth --depth_constraint

5. 组合训练
python train.py -s data/369/chest_50_3views.pickle -m output/combined \
    --multi_gaussian --pseudo_labels --depth_constraint \
    --num_additional_views 15 \
    --multi_gaussian_weight 0.15 \
    --pseudo_label_weight 0.08 \
    --depth_loss_weight 0.12

6. 自定义参数训练
python train.py -s data/369/chest_50_3views.pickle -m output/custom \
    --multi_gaussian --pseudo_labels \
    --num_additional_views 20 \
    --pseudo_confidence_threshold 0.9 \
    --multi_gaussian_weight 0.2 \
    --pseudo_label_weight 0.1

7. 使用Drop方法训练
python train.py -s data/369/chest_50_3views.pickle -m output/with_drop \
    --enable_drop --drop_rate 0.10
"""

import os
import sys
import subprocess

def run_training_example():
    """运行训练示例"""
    
    # 基础训练
    print("=== 基础训练示例 ===")
    cmd_basic = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/basic_example"
    ]
    print("命令:", " ".join(cmd_basic))
    print("说明: 使用原始r2-gaussian功能训练")
    print()
    
    # 多高斯训练
    print("=== 多高斯训练示例 ===")
    cmd_multi = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/multi_example",
        "--multi_gaussian",
        "--num_additional_views", "10",
        "--multi_gaussian_weight", "0.1"
    ]
    print("命令:", " ".join(cmd_multi))
    print("说明: 启用多高斯训练，生成10个额外视角")
    print()
    
    # 伪标签训练
    print("=== 伪标签训练示例 ===")
    cmd_pseudo = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/pseudo_example",
        "--pseudo_labels",
        "--num_additional_views", "8",
        "--pseudo_label_weight", "0.05",
        "--pseudo_confidence_threshold", "0.8"
    ]
    print("命令:", " ".join(cmd_pseudo))
    print("说明: 启用伪标签训练，使用置信度阈值0.8")
    print()
    
    # 组合训练
    print("=== 组合训练示例 ===")
    cmd_combined = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/combined_example",
        "--multi_gaussian",
        "--pseudo_labels",
        "--depth_constraint",
        "--num_additional_views", "15",
        "--multi_gaussian_weight", "0.15",
        "--pseudo_label_weight", "0.08",
        "--depth_loss_weight", "0.12"
    ]
    print("命令:", " ".join(cmd_combined))
    print("说明: 同时启用多高斯、伪标签、深度约束训练")
    print()
    
    # 高级训练
    print("=== 高级训练示例 ===")
    cmd_advanced = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/advanced_example",
        "--multi_gaussian",
        "--pseudo_labels",
        "--num_additional_views", "20",
        "--pseudo_confidence_threshold", "0.9",
        "--multi_gaussian_weight", "0.2",
        "--pseudo_label_weight", "0.1",
        "--iterations", "50000",  # 增加训练轮数
        "--test_iterations", "10000,20000,30000,40000,50000"
    ]
    print("命令:", " ".join(cmd_advanced))
    print("说明: 高级训练配置，增加训练轮数和测试频率")
    print()

    # 使用 Drop 方法训练（可配置丢弃比例）
    print("=== 使用 Drop 方法训练示例 ===")
    cmd_drop = [
        "python", "train.py",
        "-s", "data/369/chest_50_3views.pickle",
        "-m", "output/with_drop_example",
        "--enable_drop",
        "--drop_rate", "0.10",
        "--iterations", "2000"
    ]
    print("命令:", " ".join(cmd_drop))
    print("说明: 启用 Drop，丢弃比例 10%，训练 2000 轮（每500轮打印一次）")
    print()

def show_feature_comparison():
    """显示功能对比"""
    print("=== 功能对比表 ===")
    print("| 功能 | 原始r2-gaussian | 增强版r2-gaussian |")
    print("|------|----------------|-------------------|")
    print("| 基础训练 | ✅ | ✅ |")
    print("| 多高斯训练 | ❌ | ✅ |")
    print("| 伪标签训练 | ❌ | ✅ |")
    print("| 深度约束 | ❌ | ✅ |")
    print("| 组合训练 | ❌ | ✅ |")
    print("| 向后兼容 | ✅ | ✅ |")
    print()

def show_parameter_description():
    """显示参数说明"""
    print("=== 新增参数说明 ===")
    print("--multi_gaussian: 启用多高斯训练")
    print("--pseudo_labels: 启用伪标签训练")
    print("--depth_constraint: 启用深度约束")
    print("--num_additional_views: 额外视角数量 (默认: 10)")
    print("--pseudo_confidence_threshold: 伪标签置信度阈值 (默认: 0.8)")
    print("--multi_gaussian_weight: 多高斯损失权重 (默认: 0.1)")
    print("--pseudo_label_weight: 伪标签损失权重 (默认: 0.05)")
    print("--depth_loss_weight: 深度损失权重 (默认: 0.1)")
    print("--enable_drop: 启用随机丢弃高斯点以做正则化")
    print("--drop_rate: 丢弃比例 (0~1，默认: 0.10)")
    print()

if __name__ == "__main__":
    print("R2-Gaussian增强版使用指南")
    print("=" * 50)
    print()
    
    show_feature_comparison()
    show_parameter_description()
    run_training_example()
    
    print("=== 注意事项 ===")
    print("1. 确保在r2_ours目录下运行")
    print("2. 确保conda环境已激活: conda activate r2_gaussian_new")
    print("3. 多高斯和伪标签功能会增加训练时间")
    print("4. 建议先测试基础功能，再逐步启用高级功能")
    print("5. 可以根据具体数据调整参数权重")
    print()
    
    print("=== 性能建议 ===")
    print("- 小数据集: num_additional_views = 5-10")
    print("- 大数据集: num_additional_views = 10-20")
    print("- 快速测试: 只启用multi_gaussian")
    print("- 高质量训练: 组合使用所有功能")
    print("- 内存不足: 减少num_additional_views")
