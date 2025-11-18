#!/bin/bash
# FSGS 优化实验 - 修复医学约束和过度密化问题
# 实验名称: 2025_11_18_foot_3views_fsgs_fixed_v2
# 修复内容:
#   1. ✅ 启用医学约束 (enable_medical_constraints=True)
#   2. ✅ 修复opacity索引越界bug
#   3. ✅ 提高densify_grad_threshold (2e-4，避免过度密化)
#   4. ✅ 提高proximity_threshold (8.0，更严格的密化条件)
#   5. ✅ 增加k_neighbors (6，更准确的proximity计算)

# 初始化conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

python train.py \
  -s data/369/foot_50_3views.pickle \
  -m output/2025_11_18_foot_3views_fsgs_fixed_v2 \
  --iterations 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  --save_iterations 30000 \
  --eval \
  \
  --gaussiansN 2 \
  --coreg \
  --coprune \
  --coprune_threshold 5 \
  --normal \
  --pseudo_strategy single \
  --sample_method uniform \
  --add_num 50 \
  \
  --multi_gaussian \
  --pseudo_labels \
  --num_additional_views 50 \
  --pseudo_confidence_threshold 0.8 \
  --multi_gaussian_weight 0.05 \
  --pseudo_label_weight 0.05 \
  \
  --enable_fsgs_proximity \
  --proximity_threshold 8.0 \
  --proximity_k_neighbors 6 \
  --proximity_organ_type foot \
  \
  --enable_fsgs_depth \
  --fsgs_depth_weight 0.05 \
  --enable_fsgs_pseudo_views \
  --num_fsgs_pseudo_views 10 \
  --fsgs_start_iter 2000 \
  \
  --densify_grad_threshold 2.0e-4 \
  --densify_until_iter 12000 \
  --max_num_gaussians 200000 \
  \
  --lambda_tv 0.08

echo ""
echo "✅ FSGS 优化实验已启动"
echo "   实验目录: output/2025_11_18_foot_3views_fsgs_fixed_v2"
echo ""
echo "核心修复:"
echo "  - 医学约束已启用（默认True）"
echo "  - Opacity索引bug已修复"
echo "  - 密化阈值提高4倍（2e-4 vs 5e-5）"
echo "  - Proximity阈值提高（8.0 vs 6.0）"
echo ""
echo "预期效果:"
echo "  - 测试集PSNR: 28.24 → 29.5~30.5 dB (+1.3~2.3 dB)"
echo "  - 泛化差距: 25.79 → 18~22 dB (-4~8 dB)"
echo "  - 高斯点数: 11000+ → 8000~9000 (-20~30%)"
echo ""
