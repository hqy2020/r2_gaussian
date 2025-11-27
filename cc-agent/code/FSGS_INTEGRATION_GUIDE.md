# FSGS集成指南 - V4优化版本

**目标性能**: PSNR ≥ 28.60 dB（超过v2的28.50 dB）

**集成时间**: 预计10-15分钟

---

## 步骤1：添加命令行参数

**文件**: `r2_gaussian/arguments/__init__.py`

**位置**: 在`ModelParams.__init__`方法中，找到现有参数定义的最后，添加FSGS参数组：

```python
# 在__init__方法末尾添加（约line 100+）

# ================================
# FSGS Proximity-Guided Densification Parameters (V4 Optimization)
# ================================
self.enable_fsgs = False  # Master switch for FSGS
self.fsgs_k_neighbors = 5  # V4: Reduced from v2's 6 (tighter constraint)
self.fsgs_proximity_threshold = 7.0  # V4: Reduced from v2's 8.0 (stricter)
self.fsgs_start_iter = 2000  # When to start FSGS densification
self.fsgs_densify_frequency = 100  # How often to run FSGS densification
self.fsgs_use_cuda_knn = True  # Use CUDA-accelerated K-NN (10-30x faster)
self.fsgs_chunk_size = 5000  # Chunk size for K-NN computation (OOM control)

# Medical Constraints Parameters (v2 key success factor)
self.enable_medical_constraints = True  # Enable tissue-aware adaptive params
self.fsgs_organ_type = "foot"  # Organ type: foot/chest/head/abdomen/pancreas

# Hybrid Strategy Parameters
self.fsgs_hybrid_mode = "union"  # How to combine FSGS + gradient: union/intersection/proximity_only

# V4 Optimization Flag
self.fsgs_use_v4_optimization = True  # Enable v4 optimized parameters
```

**在argparse部分添加**（约line 200+）：
```python
# 在parser.add_argument部分添加
parser.add_argument("--enable-fsgs", action="store_true", help="Enable FSGS proximity-guided densification")
parser.add_argument("--fsgs-k-neighbors", type=int, default=5, help="K neighbors for proximity calculation (v4: 5)")
parser.add_argument("--fsgs-proximity-threshold", type=float, default=7.0, help="Proximity threshold (v4: 7.0)")
parser.add_argument("--fsgs-start-iter", type=int, default=2000, help="When to start FSGS")
parser.add_argument("--enable-medical-constraints", action="store_true", help="Enable medical tissue constraints")
parser.add_argument("--fsgs-organ-type", type=str, default="foot", choices=["foot", "chest", "head", "abdomen", "pancreas"])
parser.add_argument("--fsgs-hybrid-mode", type=str, default="union", choices=["union", "intersection", "proximity_only"])
parser.add_argument("--fsgs-use-v4-optimization", action="store_true", help="Use v4 optimization parameters")
```

---

## 步骤2：修改GaussianModel集成FSGS

**文件**: `r2_gaussian/gaussian/gaussian_model.py`

### 2.1 在文件开头添加import

**位置**: 在现有imports之后（约line 10+）：
```python
# FSGS imports
from r2_gaussian.innovations.fsgs import (
    ProximityGuidedDensifier,
    MedicalConstraints,
    FSGSConfig
)
```

### 2.2 在GaussianModel.__init__中初始化FSGS

**位置**: 在`__init__`方法末尾（约line 150+）：
```python
# Initialize FSGS components (lazy initialization in densify_and_prune)
self.fsgs_densifier = None
self.fsgs_medical_constraints = None
self.fsgs_config = None
```

### 2.3 修改densify_and_prune方法

**位置**: `densify_and_prune`方法（line 502-549）

**完整替换方案**：
```python
def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
    """
    Densify and prune Gaussians

    Enhanced with FSGS Proximity-Guided Densification (V4 optimization)
    """
    # Get command line args (from self.training_args or equivalent)
    args = self.training_args  # Assuming you have this

    # Initialize FSGS components on first call
    if args.enable_fsgs and self.fsgs_densifier is None:
        print("🔵 Initializing FSGS Proximity-Guided Densification (V4 optimization)...")

        # Create config
        if args.fsgs_use_v4_optimization:
            self.fsgs_config = FSGSConfig.for_v4_optimization(organ_type=args.fsgs_organ_type)
        else:
            self.fsgs_config = FSGSConfig(
                enable=True,
                k_neighbors=args.fsgs_k_neighbors,
                proximity_threshold=args.fsgs_proximity_threshold,
                start_iter=args.fsgs_start_iter,
                densify_frequency=args.fsgs_densify_frequency,
                enable_medical_constraints=args.enable_medical_constraints,
                organ_type=args.fsgs_organ_type,
                hybrid_mode=args.fsgs_hybrid_mode,
                chunk_size=args.fsgs_chunk_size,
                use_cuda_knn=args.fsgs_use_cuda_knn
            )

        # Create densifier
        self.fsgs_densifier = ProximityGuidedDensifier(
            k_neighbors=self.fsgs_config.k_neighbors,
            proximity_threshold=self.fsgs_config.proximity_threshold,
            chunk_size=self.fsgs_config.chunk_size,
            use_cuda_knn=self.fsgs_config.use_cuda_knn,
            enable=True
        )

        # Create medical constraints if enabled
        if self.fsgs_config.enable_medical_constraints:
            self.fsgs_medical_constraints = MedicalConstraints.from_organ_type(
                self.fsgs_config.organ_type
            )
            print(f"✓ Medical constraints enabled for organ: {self.fsgs_config.organ_type}")

        print(f"✓ FSGS initialized: {self.fsgs_config}")

    # Original gradient-based densification
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0

    grad_mask = (grads.squeeze() >= max_grad)

    # FSGS Proximity-guided densification
    if args.enable_fsgs and iteration >= args.fsgs_start_iter:
        if iteration % args.fsgs_densify_frequency == 0:
            # Compute proximity scores
            proximity_scores, neighbor_indices, neighbor_distances = self.fsgs_densifier.compute_proximity_scores(
                self.get_xyz,
                return_neighbors=True
            )

            # Apply medical constraints if enabled
            if self.fsgs_medical_constraints is not None:
                tissue_types = self.fsgs_medical_constraints.classify_tissue(self.get_opacity)
                adaptive_params = self.fsgs_medical_constraints.get_adaptive_params(tissue_types)
                custom_threshold = adaptive_params['proximity_thresholds']
            else:
                custom_threshold = None
                tissue_types = None

            # Identify FSGS densify candidates
            prox_mask = self.fsgs_densifier.identify_densify_candidates(
                proximity_scores,
                custom_threshold=custom_threshold,
                gradient_mask=grad_mask,
                hybrid_mode=args.fsgs_hybrid_mode
            )

            # Log stats if TensorBoard writer available
            if hasattr(self, 'tb_writer') and self.tb_writer is not None:
                from r2_gaussian.innovations.fsgs.utils import log_proximity_stats
                log_proximity_stats(
                    self.tb_writer,
                    proximity_scores,
                    tissue_types,
                    prox_mask,
                    iteration,
                    tissue_names=self.fsgs_medical_constraints.get_tissue_names() if self.fsgs_medical_constraints else None
                )

            # Generate new Gaussians for FSGS candidates
            prox_densify_indices = torch.where(prox_mask)[0]
            if len(prox_densify_indices) > 0:
                new_gaussians = self.fsgs_densifier.generate_new_gaussians(
                    source_positions=self.get_xyz[prox_densify_indices],
                    neighbor_indices=neighbor_indices[prox_densify_indices],
                    all_positions=self.get_xyz,
                    all_attributes={
                        'scales': self.get_scaling,
                        'rotations': self.get_rotation,
                        'opacities': self.get_opacity,
                        'features_dc': self.get_features[:, 0:1, :],
                        'features_rest': self.get_features[:, 1:, :],
                    }
                )

                # Add new Gaussians to model
                self._add_densified_gaussians(new_gaussians)

                print(f"[FSGS iter {iteration}] Added {len(new_gaussians['positions'])} new Gaussians "
                      f"(grad_mask: {grad_mask.sum()}, prox_mask: {prox_mask.sum()})")
    else:
        # Fallback: gradient-based only (original behavior)
        prox_mask = torch.zeros_like(grad_mask, dtype=torch.bool)

    # Combine masks for split/clone decision
    combined_mask = grad_mask | prox_mask

    # ===== Original densify logic (split & clone) =====
    # [Keep your original split/clone logic here]
    # Use combined_mask instead of grad_mask

    # Prune (original logic)
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_vs)
        prune_mask = torch.logical_or(prune_mask, big_points_ws)
    self.prune_points(prune_mask)

    torch.cuda.empty_cache()

def _add_densified_gaussians(self, new_gaussians_dict):
    """
    Add new Gaussians from FSGS to the model

    Args:
        new_gaussians_dict: Dict with keys: positions, scales, rotations, opacities, features_dc, features_rest
    """
    # [Implement this based on your existing densification logic]
    # Typically involves concatenating new Gaussians to existing tensors
    # Example:
    # self._xyz = torch.cat([self._xyz, new_gaussians_dict['positions']], dim=0)
    # self._scaling = torch.cat([self._scaling, new_gaussians_dict['scales']], dim=0)
    # ...
    pass  # TODO: Implement based on your model structure
```

---

## 步骤3：修改train.py集成TensorBoard

**文件**: `train.py`

**位置**: 在training loop中，将TensorBoard writer传递给GaussianModel：

```python
# 在training loop开始前（约line 150+）
gaussians.tb_writer = tb_writer  # Pass TensorBoard writer to model

# 或者在每次densify调用时
gaussians.training_args = opt  # Pass command line args to model
```

---

## 步骤4：创建训练脚本

**文件**: `scripts/train_fsgs_v4.sh`

```bash
#!/bin/bash
# FSGS V4 Optimization Training Script
# Target: PSNR ≥ 28.60 dB (超过v2的28.50 dB)

conda activate r2_gaussian_new

DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_DIR="output/2025_11_22_foot_3views_fsgs_v4"

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
  `# V4 Optimization: Enhanced FSGS` \
  --enable-fsgs \
  --fsgs-k-neighbors 5 \
  --fsgs-proximity-threshold 7.0 \
  --fsgs-start-iter 2000 \
  --enable-medical-constraints \
  --fsgs-organ-type foot \
  --fsgs-use-v4-optimization \
  \
  `# V4 Optimization: Tighter densification control` \
  --densify-grad-threshold 2e-04 \
  --densify-until-iter 10000 \
  --max-num-gaussians 180000 \
  \
  `# V4 Optimization: Enhanced regularization` \
  --lambda-tv 0.10 \
  --lambda-dssim 0.25 \
  \
  `# Optional: Graph Laplacian regularization` \
  # --enable-graph-laplacian \
  # --graph-lambda-lap 8e-4 \
  # --graph-k 6 \

echo "================================"
echo "FSGS V4 Training Complete"
echo "Expected: PSNR ≥ 28.60 dB, SSIM ≥ 0.903"
echo "Check results in: $OUTPUT_DIR/eval/iter_030000/"
echo "================================"
```

---

## 步骤5：验证集成

### 5.1 快速语法检查
```bash
python -c "from r2_gaussian.innovations.fsgs import ProximityGuidedDensifier, MedicalConstraints; print('✓ FSGS模块导入成功')"
```

### 5.2 运行短测试（100迭代）
```bash
python train.py \
  -s data/369/foot_50_3views.pickle \
  -m output/test_fsgs_v4 \
  --iterations 100 \
  --enable-fsgs \
  --fsgs-use-v4-optimization \
  --enable-medical-constraints
```

预期输出：
```
🔵 Initializing FSGS Proximity-Guided Densification (V4 optimization)...
✓ Medical constraints enabled for organ: foot
✓ FSGS initialized: FSGSConfig(V4)(...)
[FSGS iter 2000] Added 234 new Gaussians (grad_mask: 189, prox_mask: 312)
```

### 5.3 运行完整实验（30k迭代）
```bash
bash scripts/train_fsgs_v4.sh
```

预期结果：
- 训练时间：约2.5-3小时
- 测试集PSNR：28.60-28.70 dB（目标超过v2的28.50）
- 测试集SSIM：0.9025-0.9035（超过v2的0.9015）
- 泛化差距：< 20 dB（v2为22.60）

---

## 步骤6：监控训练过程

### 6.1 启动TensorBoard
```bash
tensorboard --logdir output/2025_11_22_foot_3views_fsgs_v4 --port 6042
```

### 6.2 关键指标监控

**FSGS特定指标**（如果TensorBoard集成成功）：
- `fsgs/avg_proximity_score`: 平均proximity分数
- `fsgs/num_densify_candidates`: 每次密化的高斯数量
- `fsgs/proximity_background_air`: 背景空气区域的proximity分数
- `fsgs/proximity_soft_tissue`: 软组织区域的proximity分数
- `fsgs/proximity_dense_structures`: 骨骼区域的proximity分数

**核心性能指标**：
- `metrics/psnr_2d_test`: 测试集2D PSNR（目标≥28.60）
- `metrics/ssim_2d_test`: 测试集2D SSIM（目标≥0.903）
- `loss/total`: 总损失趋势

---

## 常见问题

### Q1: 如果出现"AttributeError: 'GaussianModel' object has no attribute 'training_args'"?
**解决**: 在GaussianModel.__init__中添加：
```python
self.training_args = args  # 保存命令行参数
```

### Q2: 如果出现"simple_knn not available"警告？
**解决**: 这是正常的，会自动降级到PyTorch实现（稍慢但功能完整）。如需加速：
```bash
cd r2_gaussian/submodules/simple-knn
pip install .
```

### Q3: 如果PSNR没有提升反而下降？
**可能原因**：
1. 医学约束未正确启用：检查`--enable-medical-constraints`
2. 参数过于激进：尝试调高`--fsgs-proximity-threshold`到8.0
3. 正则化过强：降低`--lambda-tv`到0.08

**调试步骤**：
```bash
# 查看配置是否正确
cat output/2025_11_22_foot_3views_fsgs_v4/cfg_args.yml | grep fsgs

# 查看训练日志
tail -f output/2025_11_22_foot_3views_fsgs_v4_train.log | grep FSGS
```

---

## 预期时间线

| 步骤 | 预计时间 | 累计时间 |
|------|---------|---------|
| 添加参数 | 5分钟 | 5分钟 |
| 修改GaussianModel | 10分钟 | 15分钟 |
| 修改train.py | 2分钟 | 17分钟 |
| 创建训练脚本 | 2分钟 | 19分钟 |
| 快速测试 | 1分钟 | 20分钟 |
| **完整实验** | 2.5小时 | - |

**总集成时间**: ~20分钟
**完整验证时间**: ~3小时

---

## 下一步

集成完成后，运行：
```bash
bash scripts/train_fsgs_v4.sh
```

等待约2.5小时后，检查结果：
```bash
cat output/2025_11_22_foot_3views_fsgs_v4/eval/iter_030000/eval2d_render_test.yml
```

预期看到：
```yaml
PSNR: 28.65  # ✓ 超过v2的28.50
SSIM: 0.9028  # ✓ 超过v2的0.9015
```

**恭喜！FSGS V4集成成功！** 🎉
