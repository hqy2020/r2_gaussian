# Baseline 方法完整集成待办事项

> 目标：将 NAF、TensoRF、SAX-NeRF、X-Gaussian 四个方法完全集成到 r2_gaussian 仓库

## 当前进度

| 组件 | 状态 | 说明 |
|------|------|------|
| 目录结构 | ✅ 完成 | `r2_gaussian/baselines/` |
| registry.py | ✅ 完成 | 抽象接口定义 |
| train.py --method | ✅ 完成 | 路由逻辑 |
| 运行脚本 | ✅ 完成 | 支持新方法 |
| X-Gaussian 基础 | ✅ 完成 | model/renderer/trainer 全部实现 |
| NeRF 基础 | ✅ 完成 | network/render/trainer/encoder 全部实现 |
| CUDA 扩展 | ✅ 完成 | hashgrid 从 SAX-NeRF 复制并测试通过 |
| TensoRF 编码器 | ✅ 完成 | VM 分解实现（从 SAX-NeRF-master 复制）|
| Lineformer 网络 | ✅ 完成 | 线段注意力实现（从 SAX-NeRF-master 复制）|

---

## ✅ 已完成任务

### X-Gaussian 集成
- [x] 修复渲染器参数名（`opacities` not `densities`）
- [x] 添加相机 mode 支持（平行/透视投影）
- [x] 修复 import 路径（inverse_sigmoid 等）
- [x] 添加 query_xgaussian 函数
- [x] 测试训练通过（100 iter, psnr3d 21.38, psnr2d 22.42）

### NeRF 系列集成
- [x] 从 SAX-NeRF 复制 hashencoder CUDA 扩展
- [x] 实现 HashEncoder 带 JIT 编译
- [x] 实现 generate_rays_from_camera 函数
- [x] 实现完整的 trainer.py
- [x] NAF 测试通过（100 iter, psnr3d 17.82）

### TensoRF 完整实现 (2025-12-12)
- [x] 复制 SAX-NeRF-master 的 VM 分解编码器
- [x] 实现 `init_svd_volume()` 和 `compute_densityfeature()`
- [x] 更新 TensoRFConfig 添加 `density_n_comp` 和 `app_dim` 参数
- [x] 更新编码器工厂以正确传递参数

### SAX-NeRF (Lineformer) 完整实现 (2025-12-12)
- [x] 复制 SAX-NeRF-master 的 Lineformer 网络
- [x] 实现 `LineAttention`, `FFN`, `Line_Attention_Blcok` 等类
- [x] 添加 `net_type` 配置支持
- [x] 更新 trainer.py 支持 Lineformer 网络选择

---

## ✅ 验证完成

### 第一阶段：验证迁移（P1）- 全部通过！
- [x] 测试 TensoRF 训练：**通过** (100 iter, psnr3d 17.55, ssim3d 0.686)
- [x] 测试 SAX-NeRF 训练：**通过** (100 iter, psnr3d 17.56, ssim3d 0.688)
- [x] 验证 NAF 仍然正常：**通过** (100 iter, psnr3d 17.82, ssim3d 0.468)

### 第二阶段：统一评估（P3）
- [ ] 创建统一评估脚本 `test_baselines.py`
- [ ] 运行完整对比实验
- [ ] 生成对比表格

---

## 使用方法

### X-Gaussian
```bash
./cc-agent/scripts/run_spags_ablation.sh xgaussian foot 3 0
```

### NAF
```bash
./cc-agent/scripts/run_spags_ablation.sh naf foot 3 0
```

### TensoRF
```bash
./cc-agent/scripts/run_spags_ablation.sh tensorf foot 3 0
```

### SAX-NeRF
```bash
./cc-agent/scripts/run_spags_ablation.sh saxnerf foot 3 0
```

---

## 测试结果摘要

### X-Gaussian (100 iter)
- psnr3d: 21.38, ssim3d: 0.334
- psnr2d: 22.42, ssim2d: 0.648

### NAF (100 iter) ✅
- psnr3d: 17.82, ssim3d: 0.468
- psnr2d: 14.62, ssim2d: 0.119

### TensoRF (100 iter) ✅ 新增
- psnr3d: 17.55, ssim3d: 0.686
- 使用 VM 分解编码器

### SAX-NeRF (100 iter) ✅ 新增
- psnr3d: 17.56, ssim3d: 0.688
- 使用 Lineformer 网络 + HashGrid 编码器

*注：这是 100 次迭代的结果，完整训练需要 30000 次迭代*

---

## 迁移说明

### TensoRF 编码器
- **源文件**: `/home/qyhu/Documents/SAX-NeRF-master/src/encoder/tensorf_encoder.py`
- **目标文件**: `r2_gaussian/baselines/nerf_base/encoder/tensorf.py`
- **实现**: VM (Vector-Matrix) 分解，使用 3 个平面和 3 个向量表示 3D 特征

### Lineformer 网络
- **源文件**: `/home/qyhu/Documents/SAX-NeRF-master/src/network/Lineformer.py`
- **目标文件**: `r2_gaussian/baselines/nerf_base/lineformer.py`
- **实现**: 线段注意力机制，对射线上的采样点进行自注意力处理

---

## 文件结构

```
r2_gaussian/baselines/
├── __init__.py
├── registry.py                  # 方法注册表
├── naf/
│   ├── __init__.py
│   └── config.py               # NAFConfig
├── tensorf/
│   ├── __init__.py
│   └── config.py               # TensoRFConfig (含 density_n_comp, app_dim)
├── saxnerf/
│   ├── __init__.py
│   └── config.py               # SAXNeRFConfig (含 net_type, Lineformer 参数)
├── xgaussian/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── renderer.py
│   └── trainer.py
└── nerf_base/
    ├── __init__.py
    ├── network.py              # DensityNetwork + get_network()
    ├── lineformer.py           # Lineformer (新增)
    ├── render.py               # 渲染函数
    ├── trainer.py              # training_nerf() (支持 Lineformer)
    └── encoder/
        ├── __init__.py         # get_encoder()
        ├── frequency.py        # FreqEncoder
        ├── hashgrid.py         # HashEncoder
        ├── tensorf.py          # TensorfEncoder (VM 分解，已完成)
        └── hashencoder/        # CUDA 实现
```

---

*最后更新: 2025-12-12*
