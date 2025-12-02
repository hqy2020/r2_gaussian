#!/usr/bin/env python3
"""
Init-PCD 超参数搜索 - 点云生成脚本

使用 L9(3³) 正交表设计，9个实验覆盖3因子3水平的主效应。

超参数空间:
- n_points: [25000, 50000, 75000]
- density_thresh: [0.03, 0.05, 0.08]
- density_rescale: [0.10, 0.15, 0.20]
- sampling_strategy: density_weighted (固定)

用法:
    python scripts/init_pcd_hyperparam_search.py --generate-all
    python scripts/init_pcd_hyperparam_search.py --exp exp01 --scene foot_3
"""

import argparse
import subprocess
import sys
from pathlib import Path

# L9(3³) 正交表
# 列顺序: n_points, density_thresh, density_rescale
L9_ORTHOGONAL_ARRAY = {
    "exp01": {"n_points": 25000, "density_thresh": 0.03, "density_rescale": 0.10},
    "exp02": {"n_points": 25000, "density_thresh": 0.05, "density_rescale": 0.15},
    "exp03": {"n_points": 25000, "density_thresh": 0.08, "density_rescale": 0.20},
    "exp04": {"n_points": 50000, "density_thresh": 0.03, "density_rescale": 0.15},
    "exp05": {"n_points": 50000, "density_thresh": 0.05, "density_rescale": 0.20},
    "exp06": {"n_points": 50000, "density_thresh": 0.08, "density_rescale": 0.10},
    "exp07": {"n_points": 75000, "density_thresh": 0.03, "density_rescale": 0.20},
    "exp08": {"n_points": 75000, "density_thresh": 0.05, "density_rescale": 0.10},
    "exp09": {"n_points": 75000, "density_thresh": 0.08, "density_rescale": 0.15},
}

# 场景配置
SCENES = {
    "foot_3": {
        "pickle": "foot_50_3views.pickle",
        "init_npy": "init_foot_50_3views.npy",
    },
    "abdomen_9": {
        "pickle": "abdomen_50_9views.pickle",
        "init_npy": "init_abdomen_50_9views.npy",
    },
}

PROJECT_ROOT = Path(__file__).parent.parent
DATA_369 = PROJECT_ROOT / "data" / "369"
SEARCH_DIR = PROJECT_ROOT / "data" / "init-pcd-search"


def generate_pointcloud(exp_id: str, scene_id: str, dry_run: bool = False):
    """为指定实验和场景生成点云"""
    if exp_id not in L9_ORTHOGONAL_ARRAY:
        raise ValueError(f"Unknown experiment: {exp_id}")
    if scene_id not in SCENES:
        raise ValueError(f"Unknown scene: {scene_id}")

    params = L9_ORTHOGONAL_ARRAY[exp_id]
    scene = SCENES[scene_id]

    # 输入路径 (使用原始 369 数据)
    data_path = DATA_369 / scene["pickle"]

    # 输出路径
    output_dir = SEARCH_DIR / exp_id
    output_npy = output_dir / scene["init_npy"]

    # 构建命令 (使用 --data 参数，--output 是完整的文件路径)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "initialize_pcd.py"),
        "--data", str(data_path),
        "--output", str(output_npy),  # 完整的 .npy 文件路径
        "--n_points", str(params["n_points"]),
        "--density_thresh", str(params["density_thresh"]),
        "--density_rescale", str(params["density_rescale"]),
        "--sampling_strategy", "density_weighted",
    ]

    print(f"\n{'='*60}")
    print(f"[{exp_id}] {scene_id}")
    print(f"  n_points={params['n_points']}, thresh={params['density_thresh']}, rescale={params['density_rescale']}")
    print(f"  Output: {output_npy}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return

    # 执行
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr}")
        return False

    # 检查生成的文件（使用 --output 参数后直接生成在目标位置）
    if output_npy.exists():
        print(f"  [OK] Generated {output_npy}")
    else:
        print(f"  [ERROR] Output file not found: {output_npy}")
        return False

    return True


def generate_all(dry_run: bool = False):
    """生成所有实验配置的点云"""
    total = len(L9_ORTHOGONAL_ARRAY) * len(SCENES)
    success = 0

    print(f"Generating {total} pointcloud files ({len(L9_ORTHOGONAL_ARRAY)} experiments × {len(SCENES)} scenes)")
    print(f"Output directory: {SEARCH_DIR}")

    for exp_id in L9_ORTHOGONAL_ARRAY:
        for scene_id in SCENES:
            if generate_pointcloud(exp_id, scene_id, dry_run):
                success += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success}/{total}")

    return success == total


def print_experiment_table():
    """打印实验配置表"""
    print("\nL9(3³) 正交表实验设计:")
    print("-" * 60)
    print(f"{'Exp':<8} {'n_points':<10} {'thresh':<10} {'rescale':<10}")
    print("-" * 60)
    for exp_id, params in L9_ORTHOGONAL_ARRAY.items():
        print(f"{exp_id:<8} {params['n_points']:<10} {params['density_thresh']:<10} {params['density_rescale']:<10}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Init-PCD 超参数搜索点云生成")
    parser.add_argument("--generate-all", action="store_true", help="生成所有实验配置")
    parser.add_argument("--exp", type=str, help="指定实验ID (exp01-exp09)")
    parser.add_argument("--scene", type=str, choices=["foot_3", "abdomen_9"], help="指定场景")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--list", action="store_true", help="列出实验配置表")

    args = parser.parse_args()

    if args.list:
        print_experiment_table()
        return

    if args.generate_all:
        generate_all(args.dry_run)
    elif args.exp and args.scene:
        generate_pointcloud(args.exp, args.scene, args.dry_run)
    elif args.exp:
        # 生成指定实验的所有场景
        for scene_id in SCENES:
            generate_pointcloud(args.exp, scene_id, args.dry_run)
    else:
        parser.print_help()
        print("\n")
        print_experiment_table()


if __name__ == "__main__":
    main()
