原因分析（为什么 NeRF 系 baseline 明显落后且 3/6/9 结果“差不多”）

  - 射线几何/near-far 裁剪不正确：原 generate_rays_from_camera 用全局近远平面 + 近似距离估计，很多射线实际不穿过体素却仍被积分；
    再叠加 pts 会被 clamp 到 [-bound,bound]，导致大量采样点“挤在边界”，监督信息失真，训练容易收敛到错误/平庸解；此时加视角也帮不上
    忙，所以 3/6/9 的指标会异常接近。
  - cone-beam 探测器轴顺序潜在不一致：dDetector/offDetector 在数据里通常是 [v,u]，若按 [u,v] 用，会造成像素→探测器坐标映射错误，某
    些器官/分辨率下影响更大。
  - 优化器重复参数：你日志里出现过 “duplicate parameters” 警告，表示同一参数被重复加入优化器，训练会更不稳定，进一步放大“学不对”的
    现象。
  - 评估口径不完全一致：3DGS 默认对全部 test views 评估；NeRF baseline 训练时默认只评估前 eval_max_views 个（默认 10），会让跨方法
    对比更噪、更不稳定（但不会解释“完全不随视角变化”，那主要还是前面几条）。

  已修复的方法（代码改动摘要）

  - r2_gaussian/baselines/nerf_base/render.py:68：
      - 用 体素 AABB 与每条 ray 精确求交 计算 near/far，并对 rays_d 做归一化；不命中体素的射线 far==near，渲染自然输出 0（避免“边界
        挤压”）。
      - 修正 cone-beam 下 dDetector/offDetector 的 [v,u] 轴顺序使用。
  - r2_gaussian/baselines/nerf_base/trainer.py:263：
      - 去重优化器参数，避免重复更新同一参数。
      - 默认不再强制构建独立 fine network（需要时可用 use_fine_network 开关），同时增加 Ray-AABB hit ratio 日志用于快速自检。

  之后要做的事（建议顺序）

  - 先做 1 个器官的 smoke：用 abdomen/chest（你之前最异常的）跑 SPAGS_ITERS=2000，确认 9views > 6views > 3views 且日志里 Ray-AABB
    hit ratio 合理（不是接近 0）。
  - 再跑 5×3=15 个场景全量；统一从 output/<run_dir>/eval/iter_xxxxxx/eval2d_render_test.yml 取指标做表。
  - 若修完后仍显著落后：再考虑调参（n_samples/n_fine、学习率、eval_max_views=50 统一评估口径），而不是先怀疑视角数。

  按 cc-agent 脚本正确运行

  - 环境/数据准备：
      - conda activate r2_gaussian_new
      - 数据需在 data/369/<organ>_50_<3|6|9>views.pickle
      - 跑 3DGS（r2_gaussian/spags/xgaussian）时按脚本提示准备 init_*.npy / data/369-sps/init_*.npy（bash cc-agent/scripts/
        generate_sps_init_369.sh <gpu>）
  - 运行方式（示例）：
  
      - 单个：SPAGS_ITERS=2000 SPAGS_TEST_ITERS="2000" bash cc-agent/scripts/run_spags_ablation.sh naf abdomen 3 0 nerf_fix
      - 批量（15 场景，NAF）：for o in foot chest head abdomen pancreas; do for v in 3 6 9; do SPAGS_ITERS=2000
        SPAGS_TEST_ITERS="2000" bash cc-agent/scripts/run_spags_ablation.sh naf $o $v 0 nerf_fix; done; done
  - 指标位置：
      - output/<prefix>_<organ>_<views>views_<method>/eval/iter_00xxxx/eval2d_render_test.yml（NeRF baseline 和 3DGS 都会写这个兼容
        文件名）