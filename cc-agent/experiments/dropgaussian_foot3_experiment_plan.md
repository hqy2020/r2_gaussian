# DropGaussian Foot-3 视角消融实验方案

## 📊 实验概览

**目标**: 验证 DropGaussian 在 Foot-3 视角 CT 重建任务中的性能提升

**数据集**: `data/369/foot_50_3views.pickle`

**Baseline SOTA**:
- PSNR: 28.4873
- SSIM: 0.9005

**预期提升**:
- PSNR: +0.5~1.5 dB
- SSIM: +0.005~0.015

---

## 🧪 实验设计



### 实验 2: DropGaussian (γ=0.2, 论文推荐值)
**目的**: 验证 DropGaussian 在 CT 重建中的有效性

**训练命令**:
```bash
conda activate r2_gaussian_new

python train.py \
    -s ./data/foot_3views \
    -m ./output/2025_11_19_foot_3views_dropgaussian \
    --iterations 30000 \
    --use_drop_gaussian \
    --drop_gamma 0.2 \
    --eval
```

**配置**:
- DropGaussian: **启用**
- Drop Gamma (γ): 0.2 (论文推荐)
- Drop Rate 动态变化: 0 → 0.2 (渐进式增加)
- Iterations: 30,000
- Densification: 前 15,000 iterations
- 其他参数: 与 baseline 一致

**关键机制**:
```
iteration=0:     drop_rate = 0.2 × (0/30000) = 0.000 (无丢弃)
iteration=5000:  drop_rate = 0.2 × (5000/30000) = 0.033 (丢弃 3.3%)
iteration=10000: drop_rate = 0.2 × (10000/30000) = 0.067 (丢弃 6.7%)
iteration=15000: drop_rate = 0.2 × (15000/30000) = 0.100 (丢弃 10%)
iteration=20000: drop_rate = 0.2 × (20000/30000) = 0.133 (丢弃 13.3%)
iteration=25000: drop_rate = 0.2 × (25000/30000) = 0.167 (丢弃 16.7%)
iteration=30000: drop_rate = 0.2 × (30000/30000) = 0.200 (丢弃 20%)
```

---

### 实验 3 (可选): DropGaussian (γ=0.1, 保守设置)
**目的**: 探索更保守的丢弃策略对性能的影响

**训练命令**:
```bash
conda activate r2_gaussian_new

python train.py \
    -s ./data/foot_3views \
    -m ./output/2025_11_19_foot_3views_dropgaussian_gamma01 \
    --iterations 30000 \
    --use_drop_gaussian \
    --drop_gamma 0.1 \
    --eval
```

**配置**:
- Drop Gamma (γ): 0.1 (更保守)
- 最大丢弃率: 10%

---

## 📈 评估指标

### 主要指标
1. **PSNR** (Peak Signal-to-Noise Ratio)
   - 测量重建图像质量
   - 目标: > 29.0 dB

2. **SSIM** (Structural Similarity Index)
   - 测量结构相似性
   - 目标: > 0.905

### 次要指标
3. **Training Time**: 记录总训练时间
4. **GPU Memory**: 记录峰值 GPU 内存使用
5. **Gaussian Count**: 记录最终 Gaussian primitives 数量

---

## 🎯 成功标准

### 最低成功标准
- ✅ PSNR 提升 ≥ 0.3 dB
- ✅ SSIM 提升 ≥ 0.003

### 理想成功标准
- 🌟 PSNR 提升 ≥ 1.0 dB (达到论文水平)
- 🌟 SSIM 提升 ≥ 0.01

### 失败标准
- ❌ PSNR 下降 或 提升 < 0.1 dB
- ❌ SSIM 下降



---

## ⚡ 快速启动脚本

### 单次训练（推荐先验证）
```bash
# 激活环境
conda activate r2_gaussian_new


# DropGaussian
python train.py -s ./data/foot_3views -m ./output/2025_11_19_foot_3views_dropgaussian --iterations 30000 --use_drop_gaussian --drop_gamma 0.2 --eval
```


---

## 🔍 结果分析要点

### 定量分析
1. 对比 PSNR/SSIM 提升幅度
2. 分析训练曲线（TensorBoard）
   - Loss 收敛速度
   - PSNR 增长趋势
3. 检查 Gaussian count 变化

### 定性分析
1. 可视化对比：渲染图像质量
2. 检查伪影（artifacts）减少情况
3. 评估边缘细节保留

### 消融分析
1. 如果 γ=0.2 成功，尝试 γ=0.1 和 γ=0.3
2. 分析最优 γ 值

---

## ⚠️ 注意事项

1. **随机种子**: 使用相同的随机种子确保可复现性
2. **数据一致性**: 确保使用相同的数据集和预处理
3. **对照变量**: 除 DropGaussian 外，所有参数保持一致
5. **监控训练**: 定期检查 TensorBoard，及时发现异常

---

**实验设计日期**: 2025-11-19
**预计单次训练时间**: 2-3 小时 (30k iterations)
**预计总实验时间**: 6-9 小时 (包含 baseline + DropGaussian)
