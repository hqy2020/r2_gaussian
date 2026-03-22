# SPAGS Autoresearch

This directory contains the unattended SPAGS autoresearch tooling:

- `run_experiment.py`: run one fixed-budget experiment, evaluate, and append to `results.tsv`
- `continuous_loop.py`: repeatedly call Codex for one experiment change, commit it, run the experiment, and keep or discard the commit based on metrics
- `heartbeat.py`: collect remote loop status on the Mac, write Markdown heartbeat reports locally, and optionally push them to GitHub
- `program_medical_gs.md`: the Codex background brief for each experiment iteration

## Host requirements

- `codex` CLI installed on the host
- Codex configured in `~/.codex/config.toml`
- `r2gs` conda environment available
- dataset `/root/r2_gaussian/data/chest_50_3views.pickle`

Do not store API keys or auth tokens in this repository. Configure them on the host in `~/.codex/config.toml`.

## Startup hardening

`continuous_loop.py` now runs a Codex preflight before baseline or experiment rounds. If the configured provider/token is rejected, the loop stops immediately and writes the failure into:

- `agent_loop/loop.log`
- `agent_loop/status.json`

At runtime the loop respects the configured provider `env_key` and drops the
`OPENAI_API_KEY` fallback when a non-OpenAI auth variable is configured, so the
startup result reflects the real provider path being tested.

## Start the continuous loop

```bash
python /root/r2_gaussian/spags_autoresearch/continuous_loop.py --tag mar22
```

To stop, interrupt the process or kill the background job.

## Write a heartbeat locally

```bash
python /Users/openingcloud/IdeaProjects/PG/r2_gaussian/spags_autoresearch/heartbeat.py --push
```

This writes Markdown reports into `/Users/openingcloud/IdeaProjects/PG/reports/heartbeat/` and mirrors them into a dedicated Git worktree before pushing to `origin/main`.

The same heartbeat entrypoint is what the local hourly Codex automation should
run.
