#!/bin/bash

# 渲染 DropGaussian 的所有测试图片
conda activate r2_gaussian_new
python render.py \
  -m /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_19_foot_3views_dropgaussian \
  --iteration 30000 \
  --skip_train \
  --quiet
