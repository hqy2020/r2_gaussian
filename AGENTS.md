# Repository Guidelines
用中文输出
## 项目结构与模块组织
- `r2_gaussian/`：核心 Python 包（`gaussian/`、`dataset/`、`baselines/`、`innovations/`、`utils/`、`arguments/`）。
- 顶层入口：`train.py`（训练/方法路由）、`test.py`（评估）、`initialize_pcd.py`（SPS 初始化点云）。
- 实验脚本：`cc-agent/scripts/`（推荐用 `run_spags_ablation.sh` 统一跑消融/基线）。
- 数据与产物：`data/`、`output/`、`logs/`（均已在 `.gitignore` 中；不要提交大文件/生成物）。
- CUDA 扩展/子模块：`r2_gaussian/submodules/`（需 `--recursive` 拉取，按 `environment.yml` 进行 editable 安装）。

## 构建、测试与开发命令
- 子模块：`git submodule update --init --recursive`
- 环境：`conda env create -f environment.yml && conda activate r2_gaussian_new`
- TIGRE（数据生成/FDK 初始化）：`pip install TIGRE-2.3/Python --no-build-isolation`
- 初始化点云：`python initialize_pcd.py --data data/369/<scene>.pickle --enable_sps --n_points 50000`
- 训练（推荐脚本）：`./cc-agent/scripts/run_spags_ablation.sh spags <organ> <3|6|9> <gpu_id>`
- 训练（直接运行）：`python train.py -s data/369/<scene>.pickle -m output/<run_dir> --ply_path data/369/init_<scene>.npy`
- 评估：`python test.py -m output/<run_dir>`

## 代码风格与命名约定
- Python 3.9；4 空格缩进；仓库未统一配置 formatter/linter，保持局部风格一致即可。
- 命名：函数/变量用 `snake_case`，类用 `PascalCase`；CLI 参数集中在 `r2_gaussian/arguments/`。
- 训练输出目录建议：`yyyy_MM_dd_HH_mm_<organ>_<N>views_<technique>`。

## 测试与验证
- 本仓库暂无正式单测；改动后至少进行一次小规模 smoke run（少迭代）并运行 `python test.py`。
- 若修改 `r2_gaussian/submodules/`，需确认在 `r2_gaussian_new` 环境中可成功编译/导入。

## 提交与 PR 指南
- 提交信息保持简短明确（历史提交多为中文）；需要时加范围前缀（如 `gar:`、`adm:`、`sps:`）。
- PR 需包含：改动摘要、可复现实验命令、关键指标/截图；不要提交生成物（`data/`、`output/`、`*.pickle`、`*.npy`、日志）。

## 配置提示
- `environment.yml` 固定了 CUDA/PyTorch 版本；如需升级依赖或改动构建流程，请同步更新并说明原因。
