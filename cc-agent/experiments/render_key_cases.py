#!/usr/bin/env python3
"""
渲染关键测试案例：Top 5 Good Cases 和 Top 5 Fail Cases
"""
import sys
import os
import subprocess

# 关键案例
top_good_cases = [46, 33, 32, 45, 34]  # Top 5 最大提升
top_fail_cases = [26, 17, 18, 3, 2]     # Top 5 最大下降

print("="*80)
print("🖼️ 渲染关键测试案例")
print("="*80)

# 渲染 DropGaussian 的关键案例
print("\n📌 准备渲染 DropGaussian 模型的关键案例...")
drop_output = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_19_foot_3views_dropgaussian"
drop_render_dir = f"{drop_output}/eval/iter_030000/render_images"

# 创建目录
os.makedirs(drop_render_dir, exist_ok=True)

print(f"\n输出目录: {drop_render_dir}")
print(f"Good Cases: {top_good_cases}")
print(f"Fail Cases: {top_fail_cases}")

# 检查 baseline 是否有渲染图片
baseline_output = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3views_r2_baseline_1113"
baseline_render_dir = f"{baseline_output}/eval/iter_030000/render_images"

if os.path.exists(baseline_render_dir):
    baseline_images = os.listdir(baseline_render_dir)
    print(f"\n✅ Baseline 已有 {len(baseline_images)} 张渲染图片")
else:
    print(f"\n⚠️ Baseline 渲染目录不存在: {baseline_render_dir}")
    print("需要先渲染 Baseline")

print("\n" + "="*80)
print("📝 渲染命令（需要手动执行）")
print("="*80)

# 生成渲染命令
render_cmd = f"""
# 渲染 DropGaussian 的所有测试图片
conda activate r2_gaussian_new
python render.py \\
  -m {drop_output} \\
  --iteration 30000 \\
  --skip_train \\
  --quiet
"""

print(render_cmd)

# 保存渲染脚本
with open('/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/render_dropgaussian.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(render_cmd)

os.chmod('/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/render_dropgaussian.sh', 0o755)

print("\n✅ 渲染脚本已保存到: cc-agent/experiments/render_dropgaussian.sh")
print("\n📌 执行后，使用以下 Python 脚本对比关键案例:")
print("   python cc-agent/experiments/visualize_key_cases.py")
