# SPAGS Heartbeat — 2026-03-24 08:00 CST

- Generated at: `2026-03-24 08:46:00 CST`
- Remote: `codex/autoresearch/live20260322b` @ `157ee24`
- Loop running: `yes`
- Probe status: `ok`
- Probe port: `23`
- Loop state: `running`
- Round idx: `114`
- Last decision: `testing`

## 本次结论

- 旧 autoresearch 账本新增 `92` 条记录，累计 `92` 条。
- 当前仍只能从旧账本确认最佳 commit=`131049b`，psnr=`26.362200`，ssim=`0.839783`。
- 旧账本最近 `3` 条结果概览：discard=3。
- 方法矩阵进度维持在 `0/75`（`0.0%`）。
- 当前正在测试 `157ee24`：`autoresearch: back off low-overlap pseudo supervision`。
- 研究总目标按“5×5×3 方法矩阵”共 `75` 个矩阵 job 计，当前完成 `0/75`（`0.0%`）。

## 目标进度

- 目标：`5×5×3 方法矩阵`
- 进度：`0/75` (`0.0%`)

## SPAGS 快基准

- Candidate rows: `0`
- Best candidate commit: `n/a`
- Best candidate psnr: `n/a`
- Best candidate ssim: `n/a`

## 方法矩阵

- Completed jobs: `0/75`
- Status counts: `{}`

## 审计状态

- Audit signature: `None`
- High-risk findings: `0`

## 当前运行态

- Preflight status: `success`
- Preflight summary: `OK`
- Preflight exit code: `0`
- Preflight at: `2026-03-22T14:40:31Z`

## 指标摘要

- Ledger exists: `yes`
- Experiment rows: `92`
- Current testing: `157ee24` desc=`autoresearch: back off low-overlap pseudo supervision`
- Best keep: `131049b` psnr=`26.362200` ssim=`0.839783` desc=`autoresearch: back off zero-overlap pseudo supervision`

### Candidate Tail

| candidate_id | commit | psnr_2d | ssim_2d | status | conclusion |
|---|---|---:|---:|---|---|
| - | - | - | - | - | - |

### Last 3 Rows

| commit | psnr_2d | ssim_2d | status | description |
|---|---:|---:|---|---|
| 942983d | 26.356962 | 0.838948 | discard | autoresearch: add gradient loss warmup |
| 72b5b21 | 26.356962 | 0.838948 | discard | autoresearch: reuse pseudo depth volume query |
| d479906 | 26.356962 | 0.838948 | discard | autoresearch: confidence-weight pseudo supervision |

## 调试附录


### Loop Log Tail

```text
[2026-03-23 22:24:54] Round 108: starting from 131049b
[2026-03-23 22:35:36] Round 108: testing commit 8e9f04c autoresearch: normalize pseudo overlap weight
[2026-03-23 22:50:52] Round 108: discard 8e9f04c and reset HEAD~1
[2026-03-23 22:50:57] Round 109: starting from 131049b
[2026-03-23 22:59:07] Round 109: testing commit c7d5eac autoresearch: add scale-density regularizer
[2026-03-23 23:14:23] Round 109: discard c7d5eac and reset HEAD~1
[2026-03-23 23:14:28] Round 110: starting from 131049b
[2026-03-23 23:16:59] Round 110: testing commit 81e937b autoresearch: apply dssim loss per gaussian
[2026-03-23 23:32:14] Round 110: discard 81e937b and reset HEAD~1
[2026-03-23 23:32:19] Round 111: starting from 131049b
[2026-03-23 23:39:35] Round 111: testing commit 942983d autoresearch: add gradient loss warmup
[2026-03-23 23:54:51] Round 111: discard 942983d and reset HEAD~1
[2026-03-23 23:54:56] Round 112: starting from 131049b
[2026-03-23 23:59:46] Round 112: testing commit 72b5b21 autoresearch: reuse pseudo depth volume query
[2026-03-24 00:15:03] Round 112: discard 72b5b21 and reset HEAD~1
[2026-03-24 00:15:08] Round 113: starting from 131049b
[2026-03-24 00:21:15] Round 113: testing commit d479906 autoresearch: confidence-weight pseudo supervision
[2026-03-24 00:36:30] Round 113: discard d479906 and reset HEAD~1
[2026-03-24 00:36:35] Round 114: starting from 131049b
[2026-03-24 00:42:19] Round 114: testing commit 157ee24 autoresearch: back off low-overlap pseudo supervision
```
