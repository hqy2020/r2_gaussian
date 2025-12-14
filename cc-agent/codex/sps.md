原因分析（结合你现有 15 场景历史结果）

  - 从 output/*/eval/iter_030000/eval2d_render_test.yml 汇总看，你的 spags_3k 相比 baseline 在 15 场景平均 psnr2d 反而约 -0.18dB，
    ssim2d 仅 +0.0007；退化主要集中在 chest(3/6)、abdomen(6/9)、pancreas(3/6/9)。
  - 关键混淆项：你历史 spags_3k 的 ply_path 指向 data/369-sps-3k/*（见例如 output/2025_12_05_13_58_chest_3views_spags_3k/
    cfg_args.yml:4），而当前仓库标准 SPS init 是 data/369-sps/*；若点数/分布不一致，就不是“纯 SPS/纯 SPAGS”对比，容易出现不稳定
    结论。
  - SPS 的本质副作用：密度加权采样会让初始化点更聚集且整体密度均值更高（你现有 data/369-sps 相比 data/369 多数场景 density 均值显著
    上升），训练早期更容易出现“过密/过暗/过拟合”，对软组织场景（abdomen/pancreas）尤其敏感，从而拉低新视角 psnr2d/ssim2d。
  - 若生成 SPS init 没在 r2_gaussian_new 环境中跑，scipy 缺失会导致去噪被跳过（FDK 伪影更强），SPS 更容易被伪影引导
    （initialize_pcd.py 里会直接跳过 gaussian denoise）。

  修复方法总结（已落地到代码）

  - 解耦“采样位置”和“初始化密度”：新增 --sps_density_init_mode raw|match_valid_mean|match_valid_median，可把 SPS 采样点的
    密度缩放到与“有效体素(>density_thresh)”均值/中位数一致，并对缩放系数做裁剪，降低 SPS 导致的整体初始密度偏高与跨器官不稳
    （initialize_pcd.py:88、initialize_pcd.py:389）。
  - 更新 15 场景 SPS init 生成脚本默认超参：更保守的采样强度（uniform_ratio=0.3、gamma=0.8）+ 启用 match_valid_mean（cc-agent/
    scripts/generate_sps_init_369.sh:90）。
  - 同步文档：README.md:96、cc-agent/docs/SPS_空间先验播种.md:191 已更新到正确路径/正确命令。

  之后要做的事（按优先级）

  - 先消除混淆：确保 baseline 和 spags 用同一套 n_points（建议 50k）与同一版本代码；不要用旧的 data/369-sps-3k 和新的 data/369 混
    着比。
  - 重新生成 SPS init（15 场景）：bash ./cc-agent/scripts/generate_sps_init_369.sh <gpu_id> 50000 1（输出到 data/369-sps）。
  - 先 smoke 再 full：可临时 export SPAGS_ITERS=3000 跑完 15 场景看趋势，再回到 30000。
  - 训练与评估：用 ./cc-agent/scripts/run_spags_ablation.sh baseline <organ> <3|6|9> <gpu> 和 ./cc-agent/scripts/
    run_spags_ablation.sh spags <organ> <3|6|9> <gpu>；评估看 output/<run>/eval/iter_030000/eval2d_render_test.yml 或跑 python
    test.py -m output/<run>。
  - 若仍有个别器官退化：优先尝试把 --sps_density_init_mode 换成 match_valid_median，或进一步增大 sps_uniform_ratio / 降低
    sps_density_gamma（通常对 pancreas/abdomen 更稳）。

  如何“正确运行/使用” cc-agent 下面的文档

  - 文档就是 Markdown：在 cc-agent/docs/（例如 cc-agent/docs/SPS_空间先验播种.md:186、cc-agent/docs/GAR_几何感知细化.md、cc-agent/
    docs/SPAGS_技术架构.md）；直接用编辑器打开，或终端 less cc-agent/docs/SPS_空间先验播种.md。
  - 文档里的命令建议优先用脚本跑（它会激活 r2_gaussian_new、做 CUDA 检查并 cd 到项目根）：cc-agent/scripts/
    generate_sps_init_369.sh、cc-agent/scripts/run_spags_ablation.sh（示例见 cc-agent/docs/SPS_空间先验播种.md:191）。