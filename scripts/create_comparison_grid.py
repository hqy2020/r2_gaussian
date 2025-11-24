#!/usr/bin/env python3
"""创建四个初始化方法的并排对比图"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def create_comparison_grid(image_paths, titles, output_path):
    """创建并排对比图"""
    n_methods = len(image_paths)
    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 8))

    if n_methods == 1:
        axes = [axes]

    for idx, (img_path, title) in enumerate(zip(image_paths, titles)):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=20, fontweight='bold')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="init_comparison_test")
    args = parser.parse_args()

    image_paths = [
        f"{args.dir}/baseline_viz.png",
        f"{args.dir}/denoise_viz.png",
        f"{args.dir}/smart_viz.png",
        f"{args.dir}/combined_viz.png",
    ]

    titles = [
        "Baseline\n(原始 FDK)",
        "De-Init\n(降噪初始化)",
        "Smart Sampling\n(智能采样)",
        "Combined\n(组合方法)"
    ]

    create_comparison_grid(image_paths, titles, f"{args.dir}/comparison_all.png")
