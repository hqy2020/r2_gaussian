#!/usr/bin/env python3
"""
从已训练的 NeRF 模型生成 volume 数据用于可视化

用法:
    python scripts/generate_nerf_volumes.py --exp_dir output/_2025_12_15_16_50_chest_3views_tensorf
    python scripts/generate_nerf_volumes.py --all  # 处理所有 NeRF 实验
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append("./")

from r2_gaussian.baselines.nerf_base.trainer import NeRFModel, _query_volume_by_slices, get_method_config
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams


def load_nerf_model(exp_dir: Path, method: str):
    """加载 NeRF 模型"""
    config = get_method_config(method)
    model = NeRFModel(config).cuda()

    # 查找模型文件
    model_files = list(exp_dir.glob(f"{method}_iter_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model file found for {method} in {exp_dir}")

    # 选择最新的模型
    model_file = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    print(f"Loading model from {model_file}")

    ckpt = torch.load(model_file)
    model.net.load_state_dict(ckpt["network"])
    if model.net_fine is not None and ckpt.get("network_fine"):
        model.net_fine.load_state_dict(ckpt["network_fine"])

    return model, config


def generate_volume(exp_dir: Path, force: bool = False):
    """为单个实验生成 volume"""
    exp_dir = Path(exp_dir)

    # 读取配置
    cfg_path = exp_dir / "cfg_args.yml"
    if not cfg_path.exists():
        print(f"Skip {exp_dir}: no cfg_args.yml")
        return False

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    method = cfg.get("method", "")
    if method not in ["naf", "tensorf", "saxnerf"]:
        print(f"Skip {exp_dir}: not a NeRF method ({method})")
        return False

    # 检查输出路径
    output_dir = exp_dir / "volume"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "vol_pred.npy"

    if output_path.exists() and not force:
        print(f"Skip {exp_dir}: volume already exists")
        return True

    # 加载场景获取 scanner_cfg
    source_path = cfg.get("source_path", "")
    if not source_path or not os.path.exists(source_path):
        print(f"Skip {exp_dir}: source_path not found ({source_path})")
        return False

    # 创建临时 ModelParams
    class TempArgs:
        def __init__(self, cfg):
            self.source_path = cfg.get("source_path", "")
            self.model_path = str(exp_dir)
            self.data_device = "cuda"
            self.eval = True

    args = TempArgs(cfg)
    scene = Scene(args, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # 加载模型
    try:
        model, config = load_nerf_model(exp_dir, method)
    except FileNotFoundError as e:
        print(f"Skip {exp_dir}: {e}")
        return False

    model.eval()

    # 生成 volume
    print(f"Generating volume for {exp_dir.name}...")
    with torch.no_grad():
        vol_pred = _query_volume_by_slices(model, config, scanner_cfg)

    # 保存
    np.save(output_path, vol_pred)
    print(f"Saved volume to {output_path}")

    # 同时保存 vol_gt
    vol_gt = scene.vol_gt
    if vol_gt is not None:
        if isinstance(vol_gt, torch.Tensor):
            vol_gt_np = vol_gt.detach().cpu().numpy()
        else:
            vol_gt_np = vol_gt
        gt_path = output_dir / "vol_gt.npy"
        np.save(gt_path, vol_gt_np)
        print(f"Saved vol_gt to {gt_path}")

    return True


def find_nerf_experiments(output_dir: Path):
    """查找所有 NeRF 实验目录"""
    experiments = []
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        cfg_path = exp_dir / "cfg_args.yml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        method = cfg.get("method", "")
        if method in ["naf", "tensorf", "saxnerf"]:
            # 检查是否有 30k 模型
            model_files = list(exp_dir.glob(f"{method}_iter_30000.pth"))
            if model_files:
                experiments.append(exp_dir)
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="单个实验目录")
    parser.add_argument("--all", action="store_true", help="处理所有 NeRF 实验")
    parser.add_argument("--force", action="store_true", help="强制重新生成")
    parser.add_argument("--output_base", type=str, default="output", help="输出目录基路径")
    args = parser.parse_args()

    if args.exp_dir:
        generate_volume(Path(args.exp_dir), args.force)
    elif args.all:
        output_dir = Path(args.output_base)
        experiments = find_nerf_experiments(output_dir)
        print(f"Found {len(experiments)} NeRF experiments with 30k models")

        for exp_dir in tqdm(experiments, desc="Generating volumes"):
            try:
                generate_volume(exp_dir, args.force)
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
