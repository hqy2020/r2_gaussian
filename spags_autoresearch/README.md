# SPAGS Autoresearch

This directory contains the unattended SPAGS autoresearch tooling:

- `run_experiment.py`: run one fixed-budget experiment, evaluate, and append to `results.tsv`
- `continuous_loop.py`: repeatedly call Codex for one experiment change, commit it, run the experiment, and keep or discard the commit based on metrics
- `program_medical_gs.md`: the Codex background brief for each experiment iteration

## Host requirements

- `codex` CLI installed on the host
- Codex configured in `~/.codex/config.toml`
- `r2gs` conda environment available
- dataset `/root/r2_gaussian/data/chest_50_3views.pickle`

Do not store API keys or auth tokens in this repository. Configure them on the host in `~/.codex/config.toml`.

## Start the continuous loop

```bash
python /root/r2_gaussian/spags_autoresearch/continuous_loop.py --tag mar22
```

To stop, interrupt the process or kill the background job.
