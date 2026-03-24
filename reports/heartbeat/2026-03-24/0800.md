# SPAGS Heartbeat — 2026-03-24 08:00 CST

- Generated at: `2026-03-24 08:04:46 CST`
- Remote: `codex/autoresearch/live20260322b` @ `72b5b21`
- Loop running: `yes`
- Probe status: `ok`
- Probe port: `23`
- Loop state: `running`
- Round idx: `112`
- Last decision: `testing`

## 本次结论

- 本次相对上一次没有新增结果行，说明当前主要还在运行/观测阶段。
- 当前最佳设置仍是 `131049b`，psnr=`26.362200`，ssim=`0.839783`。
- 最近 `3` 条结果概览：discard=3。
- 当前正在测试 `72b5b21`：`autoresearch: reuse pseudo depth volume query`。
- 研究总目标按“两套 5×5×3 对比实验”共 `150` 次实验计，当前账本进度 `90/150`（`60.0%`）。
- 当前建议把 `131049b` 这组参数作为后续新视角合成可视化的候选基线。

## 目标进度

- 目标：`两套 5×5×3 对比实验`
- 进度：`90/150` (`60.0%`)

## 当前运行态

- Preflight status: `success`
- Preflight summary: `OK`
- Preflight exit code: `0`
- Preflight at: `2026-03-22T14:40:31Z`

## 指标摘要

- Ledger exists: `yes`
- Experiment rows: `90`
- Current testing: `72b5b21` desc=`autoresearch: reuse pseudo depth volume query`
- Best keep: `131049b` psnr=`26.362200` ssim=`0.839783` desc=`autoresearch: back off zero-overlap pseudo supervision`

### Last 3 Rows

| commit | psnr_2d | ssim_2d | status | description |
|---|---:|---:|---|---|
| c7d5eac | 26.356962 | 0.838948 | discard | autoresearch: add scale-density regularizer |
| 81e937b | 26.356962 | 0.838948 | discard | autoresearch: apply dssim loss per gaussian |
| 942983d | 26.356962 | 0.838948 | discard | autoresearch: add gradient loss warmup |

## 调试附录


### Loop Log Tail

```text
[2026-03-23 21:44:17] Round 106: starting from 131049b
[2026-03-23 21:49:26] Round 106: testing commit 82b9356 autoresearch: add anisotropy regularizer for gaussian scales
[2026-03-23 22:04:41] Round 106: discard 82b9356 and reset HEAD~1
[2026-03-23 22:04:46] Round 107: starting from 131049b
[2026-03-23 22:09:34] Round 107: testing commit 50a0090 autoresearch: warmup dssim weight
[2026-03-23 22:24:49] Round 107: discard 50a0090 and reset HEAD~1
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
```
