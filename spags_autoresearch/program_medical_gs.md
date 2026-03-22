# autoresearch — Medical CT 3D Gaussian Splatting (SPAGS)

This is an experiment to have Codex do iterative research on medical CT reconstruction using 3D Gaussian Splatting.

## Setup

1. Use a dedicated branch `codex/autoresearch/<tag>`.
2. Read these files for context before changing code:
   - This file
   - `train.py`
   - `test.py` (read-only)
   - `r2_gaussian/gaussian/gaussian_model.py`
   - `r2_gaussian/utils/loss_utils.py`
   - `r2_gaussian/utils/depth_utils.py`
3. Verify `/root/r2_gaussian/data/chest_50_3views.pickle` exists.
4. Treat `/root/experiments/autoresearch/results.tsv` as the experiment ledger.

## Allowed changes

- `train.py`
- `r2_gaussian/utils/**`
- `r2_gaussian/gaussian/gaussian_model.py`

## Disallowed changes

- `test.py`
- `r2_gaussian/dataset/**`
- `r2_gaussian/submodules/**`
- dependencies / package installation

## Goal

Improve both `psnr_2d` and `ssim_2d` on `chest_50_3views` under:

- fixed wall-clock budget: 15 minutes
- max iterations: 30,000

All else equal, prefer simpler changes.

## Innovation axes

### ADM — Adaptive Densification Module
- Explore view-aware densification, region-adaptive thresholds, density-guided splitting.

### SPS — Sparse-view Prior Synthesis
- Explore better pseudo-view generation, geometric consistency, uncertainty-weighted pseudo supervision.

### GAR — Geometry-Aware Regularization
- Explore stronger geometric priors, smoothness, local structure-aware regularization, depth-related constraints.

## Runner

The experiment runner is:

```bash
python spags_autoresearch/run_experiment.py \
  --description "short experiment summary" \
  --extra-train-args "--gaussiansN 1"
```

The runner will:

- run training with the fixed budget
- run evaluation if needed
- parse metrics
- append to `results.tsv`
- mark `keep`, `discard`, or `crash`

## Continuous loop contract

When called by the continuous loop, do exactly one code-change iteration:

1. Inspect recent results and current code.
2. Make one coherent experiment.
3. Stage and commit the code change with a message starting with `autoresearch:`.
4. Do not run training or evaluation yourself.
5. Leave the tracked worktree clean when you exit.
