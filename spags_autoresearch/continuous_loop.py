#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time


def sh(cmd, cwd, check=True, capture_output=True):
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def log(message, log_path=None):
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_codex_process_env():
    env = os.environ.copy()
    config_path = os.path.join(os.path.expanduser("~"), ".codex", "config.toml")
    if not os.path.exists(config_path):
        return env

    current_section = None
    assignment_re = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$')
    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                continue
            if current_section != "shell_environment_policy.set":
                continue

            match = assignment_re.match(line)
            if match:
                env[match.group(1)] = match.group(2)
    return env


def tracked_dirty(repo_root):
    result = sh(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
    )
    return bool(result.stdout.strip())


def repo_has_commit(repo_root):
    result = sh(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 0


def ensure_git_safe_directory(repo_root):
    home_dir = os.path.expanduser("~")
    result = sh(
        ["git", "config", "--global", "--get-all", "safe.directory"],
        cwd=home_dir,
        check=False,
    )
    current = {
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip()
    }
    if repo_root not in current:
        sh(
            ["git", "config", "--global", "--add", "safe.directory", repo_root],
            cwd=home_dir,
            check=True,
            capture_output=False,
        )


def ensure_git_bootstrap(repo_root, log_path=None):
    git_dir_exists = os.path.isdir(os.path.join(repo_root, ".git"))
    if git_dir_exists and repo_has_commit(repo_root):
        return

    if not git_dir_exists:
        log("No git repo found. Bootstrapping a local git repository.", log_path)
        sh(["git", "init"], cwd=repo_root, check=True, capture_output=False)
    else:
        log("Git repo exists but has no commits. Completing bootstrap.", log_path)

    sh(["git", "config", "user.name", "Codex Autoresearch"], cwd=repo_root, check=True, capture_output=False)
    sh(["git", "config", "user.email", "codex-autoresearch@local"], cwd=repo_root, check=True, capture_output=False)
    sh(["git", "add", "."], cwd=repo_root, check=True, capture_output=False)
    sh(
        ["git", "commit", "--allow-empty", "-m", "chore: bootstrap autoresearch baseline"],
        cwd=repo_root,
        check=True,
        capture_output=False,
    )


def current_branch(repo_root):
    return sh(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
    ).stdout.strip()


def current_head(repo_root):
    return sh(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
    ).stdout.strip()


def branch_exists(repo_root, branch_name):
    result = sh(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
    )
    return bool(result.stdout.strip())


def checkout_or_create_branch(repo_root, branch_name, log_path=None):
    if branch_exists(repo_root, branch_name):
        log(f"Switching to existing branch {branch_name}", log_path)
        sh(["git", "checkout", branch_name], cwd=repo_root, check=True, capture_output=False)
    else:
        start_ref = current_branch(repo_root)
        log(f"Creating branch {branch_name} from {start_ref}", log_path)
        sh(["git", "checkout", "-b", branch_name], cwd=repo_root, check=True, capture_output=False)


def read_recent_results(results_tsv, limit=8):
    if not os.path.exists(results_tsv):
        return "No results yet."
    with open(results_tsv, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if len(lines) <= 1:
        return "No experiment rows yet."
    return "\n".join(lines[-limit:])


def latest_commit_subject(repo_root):
    return sh(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=repo_root,
    ).stdout.strip()


def build_prompt(program_path, results_tsv, branch_name):
    recent_results = read_recent_results(results_tsv)
    return f"""You are performing exactly one SPAGS autoresearch iteration.

Read this file first:
- {program_path}

Current branch:
- {branch_name}

Recent results:
{recent_results}

Do exactly one coherent experiment proposal and code change.

Hard constraints:
- Modify only: train.py, r2_gaussian/utils/**, r2_gaussian/gaussian/gaussian_model.py
- Do not modify: test.py, dataset loaders, CUDA kernels, dependencies
- Do not run training or evaluation yourself
- Do not edit results.tsv
- Stage and commit all code changes before exiting
- Commit message must start with: autoresearch:
- Leave the tracked worktree clean on exit

Optimization target:
- Prefer changes that can improve both psnr_2d and ssim_2d on chest_50_3views under the fixed 15-minute budget
- Favor simple, surgical changes over large refactors
- If recent experiments suggest an axis is weak, switch axis or combine only small, plausible improvements

Before exiting:
1. Ensure there is exactly one new commit if you made a change
2. Print a short summary of what changed and why
"""


def run_codex_iteration(repo_root, program_path, results_tsv, last_message_path, model=None, env=None):
    prompt = build_prompt(program_path, results_tsv, current_branch(repo_root))
    cmd = [
        "codex",
        "exec",
        "-C",
        repo_root,
        "--full-auto",
        "--color",
        "never",
        "-c",
        "mcp_servers={}",
        "--output-last-message",
        last_message_path,
        "-",
    ]
    if model:
        cmd.extend(["-m", model])
    return subprocess.run(cmd, input=prompt, text=True, env=env)


def ensure_baseline(
    repo_root,
    runner_path,
    results_tsv,
    conda_env,
    dataset,
    model_path,
    iterations,
    time_budget_minutes,
    log_path=None,
):
    if os.path.exists(results_tsv) and os.path.getsize(results_tsv) > 0:
        with open(results_tsv, "r", encoding="utf-8") as f:
            rows = [line for line in f if line.strip()]
        if len(rows) > 1:
            return

    log("No baseline found. Running baseline experiment.", log_path)
    cmd = [
        sys.executable,
        runner_path,
        "--repo-root",
        repo_root,
        "--results-tsv",
        results_tsv,
        "--conda-env",
        conda_env,
        "--dataset",
        dataset,
        "--model-path",
        model_path,
        "--iterations",
        str(iterations),
        "--time-budget-minutes",
        str(time_budget_minutes),
        "--description",
        "baseline",
        "--extra-train-args",
        "--gaussiansN 1",
    ]
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Run SPAGS autoresearch continuously with Codex")
    parser.add_argument("--repo-root", default="/root/r2_gaussian")
    parser.add_argument("--branch-prefix", default="codex/autoresearch")
    parser.add_argument("--tag", default=dt.datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means run forever")
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--results-tsv", default="/root/experiments/autoresearch/results.tsv")
    parser.add_argument("--dataset", default="/root/r2_gaussian/data/chest_50_3views.pickle")
    parser.add_argument("--model-path", default="/root/r2_gaussian/output/autoresearch_latest")
    parser.add_argument("--conda-env", default="r2gs")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--experiment-iterations", type=int, default=30000)
    parser.add_argument("--experiment-time-budget-minutes", type=float, default=15.0)
    args = parser.parse_args()

    repo_root = args.repo_root
    if not os.path.exists(repo_root):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if shutil.which("codex") is None:
        raise RuntimeError("codex command not found in PATH")
    codex_env = load_codex_process_env()

    state_dir = os.path.join(os.path.dirname(args.results_tsv), "agent_loop")
    os.makedirs(state_dir, exist_ok=True)
    loop_log = os.path.join(state_dir, "loop.log")
    last_message_path = os.path.join(state_dir, "last_agent_message.txt")

    ensure_git_safe_directory(repo_root)
    ensure_git_bootstrap(repo_root, loop_log)
    ensure_git_safe_directory(repo_root)

    if tracked_dirty(repo_root):
        raise RuntimeError("Repo has tracked uncommitted changes. Commit or stash before starting the continuous loop.")

    branch_name = f"{args.branch_prefix}/{args.tag}"
    checkout_or_create_branch(repo_root, branch_name, loop_log)

    runner_path = os.path.join(repo_root, "spags_autoresearch", "run_experiment.py")
    program_path = os.path.join(repo_root, "spags_autoresearch", "program_medical_gs.md")
    if not os.path.exists(runner_path):
        raise RuntimeError(f"Runner not found: {runner_path}")
    if not os.path.exists(program_path):
        raise RuntimeError(f"Program file not found: {program_path}")

    ensure_baseline(
        repo_root=repo_root,
        runner_path=runner_path,
        results_tsv=args.results_tsv,
        conda_env=args.conda_env,
        dataset=args.dataset,
        model_path=args.model_path,
        iterations=args.experiment_iterations,
        time_budget_minutes=args.experiment_time_budget_minutes,
        log_path=loop_log,
    )

    round_idx = 0
    while True:
        if args.max_rounds and round_idx >= args.max_rounds:
            log("Reached max rounds. Stopping.", loop_log)
            break

        round_idx += 1
        start_head = current_head(repo_root)
        log(f"Round {round_idx}: starting from {start_head[:7]}", loop_log)

        codex_proc = run_codex_iteration(
            repo_root=repo_root,
            program_path=program_path,
            results_tsv=args.results_tsv,
            last_message_path=last_message_path,
            model=args.codex_model,
            env=codex_env,
        )
        end_head = current_head(repo_root)

        if tracked_dirty(repo_root):
            log("Tracked worktree left dirty by Codex. Resetting to HEAD.", loop_log)
            sh(["git", "reset", "--hard", "HEAD"], cwd=repo_root, check=True, capture_output=False)

        if end_head == start_head:
            log(f"Round {round_idx}: no commit produced (codex exit={codex_proc.returncode}). Skipping experiment.", loop_log)
            time.sleep(args.sleep_seconds)
            continue

        description = latest_commit_subject(repo_root)
        log(f"Round {round_idx}: testing commit {end_head[:7]} {description}", loop_log)

        experiment_cmd = [
            sys.executable,
            runner_path,
            "--repo-root",
            repo_root,
            "--results-tsv",
            args.results_tsv,
            "--conda-env",
            args.conda_env,
            "--dataset",
            args.dataset,
            "--model-path",
            args.model_path,
            "--iterations",
            str(args.experiment_iterations),
            "--time-budget-minutes",
            str(args.experiment_time_budget_minutes),
            "--description",
            description,
        ]
        experiment_proc = subprocess.run(experiment_cmd, check=False)

        if experiment_proc.returncode == 0:
            log(f"Round {round_idx}: keep {end_head[:7]}", loop_log)
        else:
            log(f"Round {round_idx}: discard {end_head[:7]} and reset HEAD~1", loop_log)
            sh(["git", "reset", "--hard", "HEAD~1"], cwd=repo_root, check=True, capture_output=False)

        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    sys.exit(main())
