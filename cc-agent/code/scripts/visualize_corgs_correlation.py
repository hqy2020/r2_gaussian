"""
CoR-GS 相关性分析可视化脚本

分析双模型 Disagreement 与重建误差的相关性,验证 CoR-GS 论文的核心假设:
- Point Disagreement 越高 → 重建质量越差
- Rendering Disagreement 越高 → 重建质量越差

用法:
    python cc-agent/code/scripts/visualize_corgs_correlation.py \
        --logdir output/foot_corgs_stage1_test \
        --output cc-agent/code/scripts/corgs_correlation_analysis.png

作者: Claude Code
日期: 2025-11-16
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_scalar(logdir, tag):
    """
    从 TensorBoard 日志读取标量数据

    Args:
        logdir: TensorBoard 日志目录
        tag: 标量标签 (如 "corgs/point_rmse")

    Returns:
        steps: 迭代步数列表
        values: 对应的值列表
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        raise ValueError(f"Tag '{tag}' not found. Available tags: {ea.Tags()['scalars']}")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return np.array(steps), np.array(values)


def compute_reconstruction_error_proxy(logdir):
    """
    计算重建误差的代理指标

    由于没有 ground truth,使用测试集 PSNR 的倒数作为误差代理:
        Error = 1 / PSNR  (PSNR 越低,误差越高)

    Args:
        logdir: TensorBoard 日志目录

    Returns:
        steps: 迭代步数
        errors: 重建误差代理值
    """
    try:
        # 尝试读取测试集 PSNR
        steps, psnr_values = load_tensorboard_scalar(logdir, "reconstruction/psnr_2d_test")
        # 转换为误差 (倒数)
        errors = 1.0 / (psnr_values + 1e-6)
        return steps, errors
    except:
        # 如果没有测试集 PSNR,使用训练损失作为代理
        try:
            steps, loss_values = load_tensorboard_scalar(logdir, "train/loss_gs0")
            return steps, loss_values
        except:
            raise ValueError("Cannot find reconstruction error metric in TensorBoard logs")


def plot_correlation(disagreement_steps, disagreement_values, error_steps, error_values,
                     disagreement_name, output_path, title):
    """
    绘制散点图并计算相关性

    Args:
        disagreement_steps: Disagreement 的迭代步数
        disagreement_values: Disagreement 值
        error_steps: 重建误差的迭代步数
        error_values: 重建误差值
        disagreement_name: Disagreement 类型名称 (如 "Point RMSE")
        output_path: 输出图像路径
        title: 图像标题
    """
    # 对齐时间步 (找到共同的迭代步数)
    common_steps = np.intersect1d(disagreement_steps, error_steps)
    if len(common_steps) == 0:
        print(f"⚠️ No common steps found between {disagreement_name} and reconstruction error")
        return

    # 提取对齐的数据
    disagreement_aligned = []
    error_aligned = []
    for step in common_steps:
        idx_d = np.where(disagreement_steps == step)[0][0]
        idx_e = np.where(error_steps == step)[0][0]
        disagreement_aligned.append(disagreement_values[idx_d])
        error_aligned.append(error_values[idx_e])

    disagreement_aligned = np.array(disagreement_aligned)
    error_aligned = np.array(error_aligned)

    # 计算 Pearson 相关系数
    correlation, p_value = stats.pearsonr(disagreement_aligned, error_aligned)

    # 线性拟合
    z = np.polyfit(disagreement_aligned, error_aligned, 1)
    p = np.poly1d(z)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(disagreement_aligned, error_aligned, alpha=0.6, s=50)
    plt.plot(disagreement_aligned, p(disagreement_aligned), "r--", linewidth=2,
             label=f'Linear Fit: y={z[0]:.4f}x+{z[1]:.4f}')

    plt.xlabel(disagreement_name, fontsize=12)
    plt.ylabel('Reconstruction Error (Proxy)', fontsize=12)
    plt.title(f'{title}\nPearson r={correlation:.4f}, p-value={p_value:.2e}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加解释文本
    corr_text = "Strong Negative" if correlation < -0.5 else "Moderate Negative" if correlation < -0.3 else "Weak"
    plt.text(0.05, 0.95, f'Correlation: {corr_text}',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✅ Saved correlation plot: {output_path}")
    print(f"   Pearson r = {correlation:.4f}, p-value = {p_value:.2e}")
    print(f"   Correlation strength: {corr_text}")

    return correlation, p_value


def main():
    parser = argparse.ArgumentParser(description="Visualize CoR-GS Disagreement vs Reconstruction Error Correlation")
    parser.add_argument("--logdir", type=str, required=True,
                        help="Path to TensorBoard log directory")
    parser.add_argument("--output", type=str, default="corgs_correlation_analysis.png",
                        help="Output image path")
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        raise FileNotFoundError(f"Log directory not found: {args.logdir}")

    print(f"Loading TensorBoard data from: {args.logdir}")

    # 1. 加载 Point Disagreement (RMSE)
    try:
        point_steps, point_rmse = load_tensorboard_scalar(args.logdir, "corgs/point_rmse")
        print(f"✅ Loaded Point RMSE: {len(point_rmse)} data points")
        has_point_rmse = True
    except Exception as e:
        print(f"⚠️ Could not load Point RMSE: {e}")
        has_point_rmse = False

    # 2. 加载 Rendering Disagreement (PSNR diff)
    try:
        render_steps, render_psnr_diff = load_tensorboard_scalar(args.logdir, "corgs/render_psnr_diff")
        print(f"✅ Loaded Rendering PSNR Diff: {len(render_psnr_diff)} data points")
        has_render_psnr = True
    except Exception as e:
        print(f"⚠️ Could not load Rendering PSNR Diff: {e}")
        has_render_psnr = False

    # 3. 加载重建误差代理指标
    try:
        error_steps, error_values = compute_reconstruction_error_proxy(args.logdir)
        print(f"✅ Loaded Reconstruction Error Proxy: {len(error_values)} data points")
    except Exception as e:
        print(f"❌ Could not load reconstruction error: {e}")
        return

    # 4. 绘制相关性图
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    results = {}

    if has_point_rmse:
        corr, pval = plot_correlation(
            point_steps, point_rmse,
            error_steps, error_values,
            "Point Disagreement (RMSE)",
            args.output.replace('.png', '_point_rmse.png'),
            "Point Disagreement vs Reconstruction Error"
        )
        results['point_rmse'] = {'correlation': corr, 'p_value': pval}

    if has_render_psnr:
        # 注意:PSNR diff 与误差应该是正相关 (PSNR diff 越高,两模型差异越大,重建越差)
        corr, pval = plot_correlation(
            render_steps, render_psnr_diff,
            error_steps, error_values,
            "Rendering Disagreement (PSNR Diff)",
            args.output.replace('.png', '_render_psnr.png'),
            "Rendering Disagreement vs Reconstruction Error"
        )
        results['render_psnr'] = {'correlation': corr, 'p_value': pval}

    # 5. 生成总结报告
    print("\n" + "=" * 60)
    print("CoR-GS 相关性分析总结")
    print("=" * 60)

    for metric, result in results.items():
        r = result['correlation']
        p = result['p_value']
        significant = "✅ 显著" if p < 0.05 else "⚠️ 不显著"
        print(f"{metric:20s}: r={r:6.4f}, p={p:.2e}  [{significant}]")

    print("\n论文假设验证:")
    if has_point_rmse and results['point_rmse']['correlation'] < -0.3:
        print("✅ Point Disagreement 与重建误差呈负相关 (符合论文预期)")
    elif has_point_rmse:
        print("⚠️ Point Disagreement 相关性弱于预期 (r > -0.3)")

    if has_render_psnr:
        # Rendering PSNR Diff 应该与误差正相关或 Point RMSE 负相关
        print("ℹ️ Rendering Disagreement 分析需结合具体数值判断")

    print("=" * 60)


if __name__ == "__main__":
    main()
