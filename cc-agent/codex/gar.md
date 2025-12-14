• 原因（为什么 GAR 不能在 3/6/9×5 器官全面提升）

  - 很多旧实验“名义上开了 GAR，实际上没发生有效密化”：当时常见配置把 proximity_threshold 设成 5.0（而你数据里邻近分数典型在
    0.01~0.1），导致候选为 0、点数长期停在 5e4，指标自然不会系统性变好（可用训练日志 pts 与 [GAR 诊断] 佐证）。
  - 后续真正开始增点的版本里，按 FSGS 思路“新点直接继承邻居 density/scale”对 CT 的射线积分渲染等价于“凭空加质量/加能量”，在 6/9
    views 更容易过拟合/噪声，出现 PSNR/SSIM 波动甚至反向。
  - 密化强度过去不随 views 变保守（until_iter/每次候选预算基本固定），导致 views 越多越可能“过密化→泛化掉分”。

  修复方法（已落地到代码）

  - 视角自适应 GAR：按训练视角数计算缩放因子，并把 effective_proximity_until_iter / gar_max_candidates / candidate_ratio_cap 随
    views 自动变保守（train.py:83、train.py:155、train.py:360）。
  - CT 稳定密化初始化：新点仍用边中点，但 rotation 继承 source；scale 可选收缩；density 做“质量守恒分摊”，同时下调 source 自身密
    度，避免“凭空加密度”导致噪声/过拟合（train.py:470）。
  - 数值稳定性：scaling_inverse_activation 加 clamp，避免尺度逆变换出现 inf/nan（r2_gaussian/gaussian/gaussian_model.py:44）。
  - 参数/脚本/文档同步：新增 GAR 稳定性开关（r2_gaussian/arguments/__init__.py:116），并把默认 gar_new_per_source 调成更稳的 1（cc-
    agent/scripts/run_spags_ablation.sh:107），README 补充新开关说明（README.md:143）。

  之后要做的事（含按 cc-agent 文档正确跑法）

  - 先确保 GPU 环境正常：conda activate r2_gaussian_new 后确认 python -c "import torch; print(torch.cuda.is_available())" 为
    True（脚本也会自动检查 CUDA）。
  - 按 cc-agent 跑 15 场景对比：用 cc-agent/scripts/run_spags_ablation.sh:1，例如
      - baseline：./cc-agent/scripts/run_spags_ablation.sh baseline foot 3 0 myrun
      - gar：./cc-agent/scripts/run_spags_ablation.sh gar foot 3 0 myrun
      - 15 场景可参考 cc-agent/scripts/run_adm_369_15.sh:1 的写法批量循环（需要的话我可以补一个 run_gar_369_15.sh /
        run_baseline_369_15.sh）。
  - 验证点：看 output/<run>/eval/iter_030000/eval2d_render_test.yml 的 psnr_2d/ssim_2d；并检查训练 log 里 [GAR 诊断] 与 pts 是否随
    迭代增长（GAR 确实在起作用）。
  - 用 cc-agent 诊断工具定位剩余问题：按 cc-agent/docs/DIAGNOSIS_TOOLS.md:1 跑 cc-agent/scripts/diagnosis/diagnose_gar.py，重点看候
    选比例是否在 5–15% 以及是否仍有过密化/密化不足。

 
