import os
import numpy as np
import imageio
from tqdm import tqdm

# === 输入目录 ===
input_image_dir = '/home/qyhu/Documents/X-Gaussian/data/abdomen_50'  # 原始 PNG 图片路径
output_root_dir = '/home/qyhu/Documents/r2_gaussian/data/cone_ntrain_50_angle_360/5_abdomen_cone'
output_proj_dir = os.path.join(output_root_dir, 'proj_train')

# === 创建输出目录 ===
os.makedirs(output_proj_dir, exist_ok=True)

# === 获取并排序 test_*.png 文件 ===
image_files = sorted(f for f in os.listdir(input_image_dir) if f.startswith('test_') and f.endswith('.png'))

# === 批量转换并保存 ===
for i, fname in tqdm(enumerate(image_files), total=len(image_files)):
    image_path = os.path.join(input_image_dir, fname)
    image = imageio.imread(image_path).astype(np.float32) / 255.0  # 转 float 并归一化
    save_path = os.path.join(output_proj_dir, f'proj_train_{i:04d}.npy')
    np.save(save_path, image)

print(f"\n✅ 已保存 {len(image_files)} 个 .npy 文件到 {output_proj_dir}")
