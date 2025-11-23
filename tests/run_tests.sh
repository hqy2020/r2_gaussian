#!/bin/bash
# GR-Gaussian 单元测试运行脚本
#
# 用法：bash tests/run_tests.sh

echo "========================================="
echo "  GR-Gaussian 单元测试"
echo "========================================="

# 激活conda环境
echo "激活 conda 环境: r2_gaussian_new"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 运行测试
echo ""
echo "运行单元测试..."
python tests/test_gr_gaussian.py

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 所有测试通过！可以进入实验阶段。"
    exit 0
else
    echo ""
    echo "❌ 测试失败！请检查错误并修复。"
    exit 1
fi
