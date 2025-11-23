#!/bin/bash
# FSGS V4 Optimization Training Script
# Target: PSNR ≥ 28.60 dB (超过v2的28.50 dB)
# Dataset: Foot-3 views
# Expected training time: ~2.5-3 hours (30k iterations)

set -e  # Exit on error

echo "================================"
echo "FSGS V4 Optimization Training"
echo "Target: PSNR ≥ 28.60 dB"
echo "================================"

# Activate conda environment
conda activate r2_gaussian_new

# Paths
DATA_PATH="data/369/foot_50_3views.pickle"
INIT_PATH="data/369/init_foot_50_3views.npy"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_fsgs_v4"

echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "================================"

# Train with FSGS V4 optimization
python train.py \
  -s $DATA_PATH \
  -m $OUTPUT_DIR \
  --port 6041 \
  --iterations 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  --save_iterations 10000 20000 30000 \
  --checkpoint_iterations 10000 20000 30000 \
  --eval \
  --views 3 \
  \
  `# V4 Optimization: Enhanced FSGS Parameters` \
  --enable-fsgs \
  --fsgs-k-neighbors 5 \
  --fsgs-proximity-threshold 7.0 \
  --fsgs-start-iter 2000 \
  --enable-medical-constraints \
  --fsgs-organ-type foot \
  --fsgs-use-v4-optimization \
  --fsgs-hybrid-mode union \
  \
  `# V4 Optimization: Tighter Densification Control` \
  --densify-grad-threshold 2.0e-4 \
  --densify-until-iter 10000 \
  --densification-interval 100 \
  --max-num-gaussians 180000 \
  \
  `# V4 Optimization: Enhanced Regularization` \
  --lambda-tv 0.10 \
  --lambda-dssim 0.25 \
  \
  `# Optional: Uncomment to enable Graph Laplacian regularization` \
  # --enable-graph-laplacian \
  # --graph-lambda-lap 8e-4 \
  # --graph-k 6 \

echo "================================"
echo "Training Complete!"
echo "End time: $(date)"
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "To check results:"
echo "  cat $OUTPUT_DIR/eval/iter_030000/eval2d_render_test.yml"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir $OUTPUT_DIR --port 6042"
echo ""
echo "Expected results:"
echo "  PSNR: 28.60-28.70 dB (vs baseline 28.49, v2 28.50)"
echo "  SSIM: 0.9025-0.9035 (vs baseline 0.9005, v2 0.9015)"
echo "================================"
