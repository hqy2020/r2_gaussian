#!/usr/bin/env python3
"""
FSGS v4+ 实验配置文件生成脚本

功能：基于 v2 baseline 配置，生成阶段 1 的 8 个实验配置文件
作者：@deep-learning-tuning-expert
创建时间：2025-11-18
"""

import yaml
import os
from pathlib import Path
from copy import deepcopy

# 配置路径
BASE_CONFIG_PATH = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_18_foot_3views_fsgs_fixed_v2/cfg_args.yml"
OUTPUT_DIR = "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/configs"

# 实验定义（8 个单因素实验）
EXPERIMENTS = [
    {
        "id": "v4_tv_0.10",
        "name": "2025_11_19_foot_3views_fsgs_v4_tv_0.10",
        "description": "强化 TV 正则化（中等强度）",
        "params": {
            "lambda_tv": 0.10
        }
    },
    {
        "id": "v4_tv_0.12",
        "name": "2025_11_19_foot_3views_fsgs_v4_tv_0.12",
        "description": "强化 TV 正则化（高强度）",
        "params": {
            "lambda_tv": 0.12
        }
    },
    {
        "id": "v4_k_5",
        "name": "2025_11_19_foot_3views_fsgs_v4_k_5",
        "description": "收紧医学约束 - 减少邻居数",
        "params": {
            "proximity_k_neighbors": 5
        }
    },
    {
        "id": "v4_tau_7.0",
        "name": "2025_11_19_foot_3views_fsgs_v4_tau_7.0",
        "description": "收紧医学约束 - 降低距离阈值",
        "params": {
            "proximity_threshold": 7.0
        }
    },
    {
        "id": "v4_densify_10k",
        "name": "2025_11_19_foot_3views_fsgs_v4_densify_10k",
        "description": "提前停止密集化",
        "params": {
            "densify_until_iter": 10000
        }
    },
    {
        "id": "v4_grad_3e-4",
        "name": "2025_11_19_foot_3views_fsgs_v4_grad_3e-4",
        "description": "提高密集化阈值（保守密集化）",
        "params": {
            "densify_grad_threshold": 3e-4
        }
    },
    {
        "id": "v4_dssim_0.30",
        "name": "2025_11_19_foot_3views_fsgs_v4_dssim_0.30",
        "description": "增强 DSSIM 权重",
        "params": {
            "lambda_dssim": 0.30
        }
    },
    {
        "id": "v4_cap_180k",
        "name": "2025_11_19_foot_3views_fsgs_v4_cap_180k",
        "description": "容量进一步限制",
        "params": {
            "max_num_gaussians": 180000
        }
    }
]


def load_base_config(config_path):
    """加载 v2 baseline 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_experiment_config(base_config, experiment):
    """生成单个实验的配置文件"""
    # 深拷贝避免修改原配置
    exp_config = deepcopy(base_config)

    # 修改 model_path
    exp_config['model_path'] = f"output/{experiment['name']}"

    # 应用参数修改
    for param_name, param_value in experiment['params'].items():
        if param_name in exp_config:
            print(f"  修改参数：{param_name} = {exp_config[param_name]} → {param_value}")
            exp_config[param_name] = param_value
        else:
            print(f"  ⚠️ 警告：参数 {param_name} 不存在于 baseline 配置中")

    # 添加实验元数据（如果配置支持）
    # exp_config['experiment_metadata'] = {
    #     'experiment_id': experiment['id'],
    #     'description': experiment['description'],
    #     'base_version': 'v2',
    #     'datetime': '2025-11-19'
    # }

    return exp_config


def save_config(config, output_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  ✅ 配置文件已保存：{output_path}")


def generate_training_script(experiments, output_path):
    """生成训练执行脚本"""
    script_lines = [
        "#!/bin/bash",
        "# FSGS v4+ 阶段 1 实验执行脚本",
        "# 自动生成于 2025-11-18",
        "",
        "set -e  # 遇到错误立即退出",
        "",
        "# 激活 conda 环境",
        "conda activate r2_gaussian_new",
        "",
        "# 检查 matplotlib bug 是否已修复",
        "echo '⚠️ 提醒：确保已修复 r2_gaussian/utils/plot_utils.py:271-274 的 matplotlib backend 问题'",
        "read -p '按 Enter 继续，或 Ctrl+C 取消...'",
        "",
        "# 定义训练函数",
        "run_experiment() {",
        "    local config_file=$1",
        "    local exp_name=$2",
        "    echo ''",
        "    echo '========================================='",
        "    echo \"开始实验：$exp_name\"",
        "    echo \"配置文件：$config_file\"",
        "    echo '========================================='",
        "    python train.py --config $config_file",
        "    echo \"✅ 实验 $exp_name 完成\"",
        "}",
        "",
        "# 执行实验",
    ]

    for exp in experiments:
        config_file = f"cc-agent/experiments/configs/{exp['id']}.yml"
        script_lines.append(f"run_experiment {config_file} {exp['id']}")

    script_lines.extend([
        "",
        "echo ''",
        "echo '========================================='",
        "echo '✅ 阶段 1 所有实验完成！'",
        "echo '========================================='",
        "echo '下一步：运行结果汇总脚本'",
        "echo '  python cc-agent/experiments/scripts/summarize_v4_results.py'",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))

    # 添加执行权限
    os.chmod(output_path, 0o755)
    print(f"✅ 训练脚本已生成：{output_path}")


def generate_parallel_script(experiments, output_path, num_gpus=4):
    """生成多 GPU 并行执行脚本"""
    # 将实验分配到不同 GPU
    gpu_assignments = {i: [] for i in range(num_gpus)}
    for idx, exp in enumerate(experiments):
        gpu_id = idx % num_gpus
        gpu_assignments[gpu_id].append(exp)

    script_lines = [
        "#!/bin/bash",
        "# FSGS v4+ 阶段 1 并行执行脚本（多 GPU）",
        "# 自动生成于 2025-11-18",
        "",
        f"# 使用 {num_gpus} 个 GPU 并行执行",
        "",
        "set -e",
        "conda activate r2_gaussian_new",
        "",
    ]

    for gpu_id, exps in gpu_assignments.items():
        script_lines.append(f"# GPU {gpu_id} 执行以下实验：")
        script_lines.append("(")
        for exp in exps:
            config_file = f"cc-agent/experiments/configs/{exp['id']}.yml"
            script_lines.append(f"    CUDA_VISIBLE_DEVICES={gpu_id} python train.py --config {config_file}")
        script_lines.append(f") &  # GPU {gpu_id} 后台运行")
        script_lines.append("")

    script_lines.extend([
        "# 等待所有后台任务完成",
        "wait",
        "echo '✅ 所有实验完成！'",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))

    os.chmod(output_path, 0o755)
    print(f"✅ 并行脚本已生成：{output_path}")


def main():
    print("=" * 60)
    print("FSGS v4+ 实验配置生成器")
    print("=" * 60)
    print()

    # 检查 baseline 配置是否存在
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"❌ 错误：找不到 baseline 配置文件：{BASE_CONFIG_PATH}")
        return

    print(f"✅ 加载 v2 baseline 配置：{BASE_CONFIG_PATH}")
    base_config = load_base_config(BASE_CONFIG_PATH)
    print(f"   原始参数：lambda_tv={base_config.get('lambda_tv')}, "
          f"k={base_config.get('proximity_k_neighbors')}, "
          f"τ={base_config.get('proximity_threshold')}")
    print()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ 配置输出目录：{OUTPUT_DIR}")
    print()

    # 生成每个实验的配置文件
    print("开始生成实验配置文件...")
    for exp in EXPERIMENTS:
        print(f"\n[{exp['id']}] {exp['description']}")
        exp_config = generate_experiment_config(base_config, exp)
        output_path = os.path.join(OUTPUT_DIR, f"{exp['id']}.yml")
        save_config(exp_config, output_path)

    print()
    print("=" * 60)
    print(f"✅ 成功生成 {len(EXPERIMENTS)} 个实验配置文件")
    print("=" * 60)
    print()

    # 生成训练执行脚本
    script_dir = "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/scripts"
    os.makedirs(script_dir, exist_ok=True)

    sequential_script = os.path.join(script_dir, "run_v4_sequential.sh")
    parallel_script = os.path.join(script_dir, "run_v4_parallel.sh")

    generate_training_script(EXPERIMENTS, sequential_script)
    generate_parallel_script(EXPERIMENTS, parallel_script, num_gpus=4)

    print()
    print("=" * 60)
    print("下一步操作：")
    print("=" * 60)
    print("1. 修复 matplotlib bug（如需要）")
    print("   文件：r2_gaussian/utils/plot_utils.py:271-274")
    print()
    print("2. 执行实验（选择一种方式）：")
    print(f"   单 GPU 顺序执行：bash {sequential_script}")
    print(f"   多 GPU 并行执行：bash {parallel_script}")
    print()
    print("3. 实验完成后汇总结果：")
    print("   python cc-agent/experiments/scripts/summarize_v4_results.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
