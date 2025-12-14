原因分析

  - 指标层面：你这批 15 场景里，ADM 对 3views 平均有小幅提升，但 6/9views 平均略回退（典型如 foot_6views、pancreas_9views），说明
    ADM 在“数据更充分”时更像扰动而不是有效先验。
  - 训练侧退化：日志里多次出现 offset≈-1 且 confidence≈1，导致 modulation 近似常数（等价于“全局密度缩放”）；这类捷径既不提供结构信
    息，还会改变 densify/prune 的阈值行为，进而伤害泛化。
  - 正则/初始化不匹配 x2gs 经验：x2gs 的 K-Planes TV 权重非常小（论文代码里 plane_tv_weight=1e-4），而你旧跑法里 TV/调制强度偏大时
    更容易把 K-Planes/decoder 推向“常数解+饱和输出”，最终变成全局缩放。
  - 评估不一致：test.py 之前没有把 ADM 的 current_iteration 对齐到加载的 checkpoint iteration，导致同一模型训练内评估和 python
    test.py 结果不一致（调度强度不同），会放大“看起来没提点/不稳定”的感觉。

  修复方法（已落地）

  - 禁止“全局缩放捷径”：把零均值从简单 mean() 改成可配置的加权零均值（默认 density*confidence），见 r2_gaussian/gaussian/
    gaussian_model.py:114、r2_gaussian/arguments/__init__.py:127（新增 --adm_zero_mean_mode）。
  - 让 6/9views 更保守：调度里把 warmup 随 views 变长、final_strength 随 views 变小，减少高视角负迁移，见 r2_gaussian/gaussian/
    gaussian_model.py:343。
  - 抑制早期饱和：把 confidence head 初始 bias 设为 -2（初始 confidence≈0.12），避免一开始就“全场强开”，见 r2_gaussian/gaussian/
    kplanes.py:231。
  - 评估对齐训练：评估侧传入完整 cfg 参数并设置 gaussians.current_iteration=loaded_iter，见 test.py:38。
  - 统一脚本参数：消融脚本补齐 --adm_zero_mean_mode density_confidence，见 cc-agent/scripts/run_spags_ablation.sh:120。

  之后要做的事（把“稳定超过 baseline”跑出来）

  - 先做 smoke（同 init/同迭代/同评估点）再全量：SPAGS_ITERS=5000 SPAGS_TEST_ITERS="1 5000" ./cc-agent/scripts/
    run_spags_ablation.sh baseline foot 6 0 与 ... adm foot 6 0；确认 ADM 日志不再饱和后再跑 30000。
  - 对比口径统一：以训练自动写出的 output/<run>/eval/iter_030000/eval2d_render_test.yml 为准；需要复核时再跑 python test.py -m
    output/<run> --iteration 30000（现在应与 eval/ 一致）。
  - 针对“还回退”的组合优先调参顺序：先减 --adm_max_range 或增 --adm_warmup_iters，再扫 --lambda_plane_tv（建议按 x2gs 经验试 1e-
    4~5e-4）。
  - 用诊断工具给“是否退化”定量：重点看 offset/confidence 是否仍接近常数、modulation 均值是否接近 1 且有方差；流程见 cc-agent/docs/
    DIAGNOSIS_TOOLS.md:1。

  如何按 cc-agent 文档正确运行

  - 文档入口：cc-agent/docs/DIAGNOSIS_TOOLS.md:1（工具怎么跑）、cc-agent/docs/AI_ASSISTED_RESEARCH_METHODOLOGY.md:90（诊断→假设→验
    证流程）。
  - 运行前固定两步：conda activate r2_gaussian_new，然后 cd /home/qyhu/Documents/r2_ours/r2_gaussian（诊断脚本用相对路径找
    PROJECT_ROOT）。
  - 训练命令以脚本为准：很多文档示例写 ./run_spags_ablation.sh，在本仓库应使用 ./cc-agent/scripts/run_spags_ablation.sh。
  - ADM/GAR/SPS 诊断与综合报告（示例）：
      - python cc-agent/scripts/diagnosis/diagnose_adm.py --checkpoint output/<run>/point_cloud/iteration_30000/point_cloud.pickle
        --output_dir diagnosis/<tag>/adm/ --adm_zero_mean --adm_view_adaptive --num_views 6
      - python cc-agent/scripts/diagnosis/analyze_training.py --baseline_dir output/<baseline_run>/ --spags_dir output/<adm_run>/
        --output_dir diagnosis/<tag>/comparison/
      - python cc-agent/scripts/diagnosis/generate_diagnosis_report.py --adm_report diagnosis/<tag>/adm/adm_diagnosis_report.json
        --training_report diagnosis/<tag>/comparison/training_analysis_report.json --output diagnosis/<tag>/full_report.md