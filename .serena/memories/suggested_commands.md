# R²-Gaussian 常用命令

## 环境管理

### 激活环境
```bash
conda activate r2_gaussian_new
```

### 查看环境信息
```bash
conda info --envs
python --version
nvcc --version
```

## 训练命令

### 标准 3 视角训练
```bash
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m /home/qyhu/Documents/r2_ours/r2_gaussian/output/footdd_3_$(date +%m%d) \
    --gaussiansN 1 \
    --enable_depth \
    --depth_loss_weight 0.05 \
    --depth_loss_type pearson \
    --pseudo_labels \
    --pseudo_label_weight 0.02 \
    --multi_gaussian_weight 0 \
    --num_additional_views 50
```

### NeRF 格式训练
```bash
python train.py -s XXX/0_chest_cone
```

### NAF 格式训练
```bash
python train.py -s XXX/*.pickle
```

## 初始化

### 生成初始化点云（FDK）
```bash
python initialize_pcd.py --data <path to data>
```

### 评估初始化质量
```bash
python initialize_pcd.py --data <path to data> --evaluate
```

### 自定义初始化参数
```bash
python initialize_pcd.py \
    --data <path to data> \
    --n_points 50000 \
    --density_thresh 0.05 \
    --density_rescale 0.15 \
    --recon_method fdk
```

## 测试与评估

### 模型评估
```bash
python test.py -m /home/qyhu/Documents/r2_ours/r2_gaussian/output/footdd_3_MMDD
```

### 生成深度图
```bash
python generate_depth_maps.py /home/qyhu/Documents/r2_ours/r2_gaussian/output/footdd_3_MMDD
```

### 绘制评估曲线
```bash
python plot_eval_curve_psnr_ssim.py
```

## TensorBoard 监控

### 启动 TensorBoard（单个实验）
```bash
tensorboard --logdir /home/qyhu/Documents/r2_ours/r2_gaussian/output/footdd_3_MMDD --port 6006
```

### 启动 TensorBoard（对比多个实验）
```bash
tensorboard --logdir /home/qyhu/Documents/r2_ours/r2_gaussian/output/ --port 6006
```

### 后台启动 TensorBoard
```bash
nohup tensorboard --logdir /home/qyhu/Documents/r2_ours/r2_gaussian/output/ --port 6006 --host 0.0.0.0 > tensorboard.log 2>&1 &
```

### SSH 端口转发访问
```bash
# 在本地机器运行
ssh -L 6006:localhost:6006 用户名@服务器IP地址
# 然后访问 http://localhost:6006
```

## 可视化

### 可视化场景
```bash
python scripts/visualize_scene.py -s <path to data>
```

### 可视化深度图
```bash
python visualize_depth_map.py
```

## 数据转换

### 转换腹部数据格式
```bash
python convert_abdomen_to_r2_format.py
```

## CUDA 扩展编译

### 编译 xray-gaussian-rasterization-voxelization
```bash
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
python setup.py install
```

### 编译 simple-knn
```bash
cd r2_gaussian/submodules/simple-knn
python setup.py install
```

## Git 操作

### 查看状态
```bash
git status
```

### 查看最近提交
```bash
git log --oneline -5
```

### 创建里程碑标签
```bash
git tag -a v1.1-add-feature-x -m "实现论文 XXX 的 YYY 功能"
git push origin v1.1-add-feature-x
```

## 系统监控

### GPU 使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

### 查看训练进程
```bash
ps aux | grep train.py
```

### 查看磁盘使用
```bash
du -sh output/*/
df -h
```

## 批处理

### 2 视角批量训练
```bash
bash batch_run_2views.sh
```

### 4 视角批量训练
```bash
bash batch_run_4views.sh
```

### FSGS 训练
```bash
bash run_fsgs_fixed.sh
```

## 项目记录命令（AI Agent）

### 记录当前工作
```bash
/record
```

### 回顾上次工作
```bash
/recap
```

### 归档 progress.md
```bash
/archive
```