# SPAGS Heartbeat — 2026-03-24 16:00 CST

- Generated at: `2026-03-24 16:11:58 CST`
- Overall status: `healthy`
- Active lane: `baseline`
- Remote: `codex/autoresearch/live20260322b` @ `181503c`
- Probe status: `ok`

## 本次结论

- Baseline 75 当前完成 `60/75`，剩余 `15` 个 job。
- SPAGS 当前已有 `4` 条 candidate 记录，其中成功 `0`、crash `3`、dry-run `1`。
- 矩阵当前平均指标领先方法是 `R2GS`，avg_psnr=`32.339448`，avg_ssim=`0.941474`。
- SPAGS 当前最佳配置是 `None` / `None`，psnr=`None`，ssim=`None`。
- SPAGS 已进入 `blocked_on_fix`，因为训练主链路相关的高风险审计发现仍有 `3` 条。
- 当前待处理队列共 `23` 项，状态分布 `{'pending': 15, 'blocked': 8}`。
- 最近一次 GitHub push 状态是 `pushed`。

## 双主线进度

- Baseline 75: `60/75` (`80.0%`)
- SPAGS SOTA: `runs=4` / `success=0` / `crash=3` / `dry_run=1`

## Baseline 75

- Remaining jobs: `15`
- Status counts: `{'completed': 60, 'failed': 1, 'missing': 14}`
- Leading method: `R2GS` / avg_psnr=`32.339448` / avg_ssim=`0.941474`

### Missing Jobs

| job_id | method | organ | views | status |
|---|---|---|---:|---|
| FSGS__chest__3views | FSGS | chest | 3 | failed |
| FSGS__chest__6views | FSGS | chest | 6 | missing |
| FSGS__chest__9views | FSGS | chest | 9 | missing |
| FSGS__foot__3views | FSGS | foot | 3 | missing |
| FSGS__foot__6views | FSGS | foot | 6 | missing |
| FSGS__foot__9views | FSGS | foot | 9 | missing |
| FSGS__head__3views | FSGS | head | 3 | missing |
| FSGS__head__6views | FSGS | head | 6 | missing |
| FSGS__head__9views | FSGS | head | 9 | missing |
| FSGS__abdomen__3views | FSGS | abdomen | 3 | missing |

## SPAGS SOTA

- Current phase: `blocked_on_fix`
- Pending candidates: `['baseline_r2gs', 'spags_sps_only', 'spags_adm_only', 'spags_gar_only', 'spags_full_balanced']`
- Audit findings: `5`
- Best candidate: `None` / `None` / psnr=`None` / ssim=`None`

### Candidate Tail

| candidate_id | commit | psnr_2d | ssim_2d | status | conclusion |
|---|---|---:|---:|---|---|
| spags_full_balanced | 37d73b8 | 0.0 | 0.0 | crash | 首个已记录候选，建立快基准基线。 |
| spags_full_balanced | 37d73b8 | 0.0 | 0.0 | crash | 候选运行崩溃，未形成有效指标；请先检查日志定位环境或训练错误。 |
| spags_full_balanced | 37d73b8 | 0.0 | 0.0 | crash | 候选运行崩溃，未形成有效指标；请先检查日志定位环境或训练错误。 |

## 当前运行态

- Loop running: `no`
- Remote loop state: `running`
- Remote last decision: `testing`
- Remote round idx: `147`

## 任务队列

- Total items: `23`
- Lane counts: `{'baseline': 15, 'spags': 8}`
- Status counts: `{'pending': 15, 'blocked': 8}`

| lane | kind | detail | status | next_retry |
|---|---|---|---|---|
| baseline | matrix_job | FSGS__chest__3views | pending | n/a |
| baseline | matrix_job | FSGS__chest__6views | pending | n/a |
| baseline | matrix_job | FSGS__chest__9views | pending | n/a |
| baseline | matrix_job | FSGS__foot__3views | pending | n/a |
| baseline | matrix_job | FSGS__foot__6views | pending | n/a |
| baseline | matrix_job | FSGS__foot__9views | pending | n/a |
| baseline | matrix_job | FSGS__head__3views | pending | n/a |
| baseline | matrix_job | FSGS__head__6views | pending | n/a |
| baseline | matrix_job | FSGS__head__9views | pending | n/a |
| baseline | matrix_job | FSGS__abdomen__3views | pending | n/a |
| baseline | matrix_job | FSGS__abdomen__6views | pending | n/a |
| baseline | matrix_job | FSGS__abdomen__9views | pending | n/a |

## Push 状态

- Last push status: `pushed`
- Last push commit: `ffa8054`
- Last push at: `2026-03-24 16:12:04 CST`

## 调试附录

### Loop Log Tail

```text
[2026-03-24 06:58:56] Round 138: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 06:59:01] Round 139: starting from 131049b
[2026-03-24 07:01:13] Round 139: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:01:18] Round 140: starting from 131049b
[2026-03-24 07:01:56] Round 140: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:02:01] Round 141: starting from 131049b
[2026-03-24 07:03:25] Round 141: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:03:30] Round 142: starting from 131049b
[2026-03-24 07:06:13] Round 142: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:06:18] Round 143: starting from 131049b
[2026-03-24 07:07:36] Round 143: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:07:41] Round 144: starting from 131049b
[2026-03-24 07:10:14] Round 144: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:10:19] Round 145: starting from 131049b
[2026-03-24 07:11:28] Round 145: no commit produced (codex exit=1 without new commit). Skipping experiment.
[2026-03-24 07:11:33] Round 146: starting from 131049b
[2026-03-24 07:20:56] Round 146: testing commit bdf712b autoresearch: downweight pseudo-warp boundary supervision
[2026-03-24 07:36:11] Round 146: discard bdf712b and reset HEAD~1
[2026-03-24 07:36:16] Round 147: starting from 131049b
[2026-03-24 07:42:13] Round 147: testing commit 7a4e96d autoresearch: add edge-aligned gradient loss
```
