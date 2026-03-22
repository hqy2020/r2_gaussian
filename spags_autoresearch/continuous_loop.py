#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time


def sh(cmd, cwd, check=True, capture_output=True, env=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message, log_path=None):
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_toml_value(raw):
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def parse_codex_config(config_path):
    data = {"": {}}
    current_section = ""
    assignment_re = re.compile(r"^([A-Za-z0-9_.-]+)\s*=\s*(.+?)\s*$")
    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                data.setdefault(current_section, {})
                continue
            match = assignment_re.match(line)
            if not match:
                continue
            data.setdefault(current_section, {})[match.group(1)] = parse_toml_value(match.group(2))
    return data


def load_codex_runtime_settings():
    config_path = os.path.join(os.path.expanduser("~"), ".codex", "config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Codex config not found: {config_path}")

    config = parse_codex_config(config_path)
    top = config.get("", {})
    provider_name = top.get("model_provider", "sub2api")
    provider_section = config.get(f"model_providers.{provider_name}", {})
    env_key = provider_section.get("env_key") or "ANTHROPIC_AUTH_TOKEN"

    return {
        "config_path": config_path,
        "disable_response_storage": bool(top.get("disable_response_storage", True)),
        "model": top.get("model"),
        "model_provider": provider_name,
        "model_reasoning_effort": top.get("model_reasoning_effort"),
        "provider": {
            "name": provider_section.get("name", provider_name),
            "base_url": provider_section.get("base_url"),
            "env_key": env_key,
            "requires_openai_auth": bool(provider_section.get("requires_openai_auth", True)),
            "wire_api": provider_section.get("wire_api", "responses"),
        },
        "shell_env": config.get("shell_environment_policy.set", {}),
    }


def quote_toml_string(value):
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_runtime_codex_config(runtime_home, settings):
    os.makedirs(runtime_home, exist_ok=True)
    provider = settings["provider"]
    config_lines = [
        f"disable_response_storage = {'true' if settings['disable_response_storage'] else 'false'}",
    ]
    if settings.get("model"):
        config_lines.append(f"model = {quote_toml_string(settings['model'])}")
    config_lines.append(f"model_provider = {quote_toml_string(settings['model_provider'])}")
    if settings.get("model_reasoning_effort"):
        config_lines.append(f"model_reasoning_effort = {quote_toml_string(settings['model_reasoning_effort'])}")
    config_lines.extend(
        [
            "",
            f"[model_providers.{settings['model_provider']}]",
            f"base_url = {quote_toml_string(provider['base_url'])}",
            f"env_key = {quote_toml_string(provider['env_key'])}",
            f"name = {quote_toml_string(provider['name'])}",
            f"requires_openai_auth = {'true' if provider['requires_openai_auth'] else 'false'}",
            f"wire_api = {quote_toml_string(provider['wire_api'])}",
            "",
        ]
    )
    with open(os.path.join(runtime_home, "config.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(config_lines))


def load_codex_process_env(settings, runtime_home):
    env = os.environ.copy()
    for key, value in settings.get("shell_env", {}).items():
        if isinstance(value, str):
            env[key] = value
    env["CODEX_HOME"] = runtime_home
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("TERM", "dumb")
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


def safe_git_output(repo_root, *args):
    result = sh(
        ["git", *args],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def current_branch(repo_root):
    value = safe_git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if value is None:
        raise RuntimeError(f"Unable to determine current branch in {repo_root}")
    return value


def current_head(repo_root):
    value = safe_git_output(repo_root, "rev-parse", "HEAD")
    if value is None:
        raise RuntimeError(f"Unable to determine current HEAD in {repo_root}")
    return value


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


def parse_results_tsv(results_tsv):
    summary = {
        "exists": os.path.exists(results_tsv),
        "row_count": 0,
        "latest_result": None,
        "best_keep": None,
    }
    if not summary["exists"]:
        return summary

    with open(results_tsv, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if len(lines) <= 1:
        return summary

    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        row = {header[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(header))}
        rows.append(row)

    summary["row_count"] = len(rows)
    summary["latest_result"] = rows[-1]
    keeps = [row for row in rows if row.get("status") == "keep"]
    if keeps:
        def metric_key(row):
            try:
                return (float(row.get("psnr_2d", 0.0)), float(row.get("ssim_2d", 0.0)))
            except Exception:
                return (0.0, 0.0)

        summary["best_keep"] = max(keeps, key=metric_key)
    return summary


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


def build_codex_cmd(repo_root, model=None, last_message_path=None):
    cmd = [
        "codex",
        "exec",
        "-C",
        repo_root,
        "--full-auto",
        "--color",
        "never",
    ]
    if model:
        cmd.extend(["-m", model])
    if last_message_path:
        cmd.extend(["--output-last-message", last_message_path])
    return cmd


def summarize_codex_failure(output_parts):
    text = "\n".join(part for part in output_parts if part).strip()
    if not text:
        return "unknown codex failure"

    interesting = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in ["401", "INVALID_API_KEY", "API_KEY_REQUIRED", "Missing environment variable", "ERROR:"]):
            interesting.append(stripped)
    if interesting:
        return " | ".join(interesting[-3:])[:500]

    last_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(last_lines[-3:])[:500]


def run_codex_preflight(repo_root, env, model=None):
    cmd = build_codex_cmd(repo_root=repo_root, model=model)
    cmd.append("Reply with OK only.")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    failure_summary = summarize_codex_failure([proc.stdout or "", proc.stderr or ""])
    success = proc.returncode == 0 and "401" not in failure_summary and "INVALID_API_KEY" not in failure_summary
    return {
        "success": success,
        "exit_code": proc.returncode,
        "summary": "OK" if success else failure_summary,
    }


def run_codex_iteration(repo_root, program_path, results_tsv, last_message_path, model=None, env=None):
    prompt = build_prompt(program_path, results_tsv, current_branch(repo_root))
    cmd = build_codex_cmd(
        repo_root=repo_root,
        model=model,
        last_message_path=last_message_path,
    )
    cmd.append("-")
    return subprocess.run(cmd, input=prompt, text=True, env=env)


def write_status(status_path, state):
    payload = dict(state)
    payload["updated_at"] = now_iso()
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def update_runtime_state(state, repo_root, results_tsv, **kwargs):
    state.update(kwargs)
    state["current_branch"] = safe_git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    state["current_head"] = safe_git_output(repo_root, "rev-parse", "HEAD")
    state["results"] = parse_results_tsv(results_tsv)


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
            return True

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
    proc = subprocess.run(cmd, check=False)
    return proc.returncode == 0


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
    parser.add_argument("--skip-codex-preflight", action="store_true")
    parser.add_argument("--status-json-path", default=None)
    args = parser.parse_args()

    repo_root = args.repo_root
    if not os.path.exists(repo_root):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if shutil.which("codex") is None:
        raise RuntimeError("codex command not found in PATH")

    default_state_dir = os.path.join(os.path.dirname(args.results_tsv), "agent_loop")
    status_json_path = args.status_json_path or os.path.join(default_state_dir, "status.json")
    state_dir = os.path.dirname(status_json_path)
    os.makedirs(state_dir, exist_ok=True)

    loop_log = os.path.join(state_dir, "loop.log")
    last_message_path = os.path.join(state_dir, "last_agent_message.txt")
    runtime_home = os.path.join(state_dir, "codex_home")

    runtime_settings = load_codex_runtime_settings()
    write_runtime_codex_config(runtime_home, runtime_settings)
    codex_env = load_codex_process_env(runtime_settings, runtime_home)

    state = {
        "pid": os.getpid(),
        "repo_root": repo_root,
        "results_tsv": args.results_tsv,
        "loop_log_path": loop_log,
        "last_agent_message_path": last_message_path,
        "status_json_path": status_json_path,
        "loop_state": "starting",
        "round_idx": 0,
        "last_decision": None,
        "last_error_summary": None,
        "last_preflight": {
            "status": "pending",
            "summary": None,
            "exit_code": None,
            "at": None,
        },
    }

    ensure_git_safe_directory(repo_root)
    ensure_git_bootstrap(repo_root, loop_log)
    ensure_git_safe_directory(repo_root)

    if tracked_dirty(repo_root):
        update_runtime_state(
            state,
            repo_root,
            args.results_tsv,
            loop_state="blocked",
            last_error_summary="tracked worktree is dirty",
        )
        write_status(status_json_path, state)
        raise RuntimeError("Repo has tracked uncommitted changes. Commit or stash before starting the continuous loop.")

    branch_name = f"{args.branch_prefix}/{args.tag}"
    checkout_or_create_branch(repo_root, branch_name, loop_log)
    update_runtime_state(state, repo_root, args.results_tsv)
    write_status(status_json_path, state)

    runner_path = os.path.join(repo_root, "spags_autoresearch", "run_experiment.py")
    program_path = os.path.join(repo_root, "spags_autoresearch", "program_medical_gs.md")
    if not os.path.exists(runner_path):
        raise RuntimeError(f"Runner not found: {runner_path}")
    if not os.path.exists(program_path):
        raise RuntimeError(f"Program file not found: {program_path}")

    if args.skip_codex_preflight:
        log("CODEX_PREFLIGHT_SKIP skip requested by flag", loop_log)
        update_runtime_state(
            state,
            repo_root,
            args.results_tsv,
            loop_state="ready",
            last_preflight={
                "status": "skipped",
                "summary": "skip requested by flag",
                "exit_code": 0,
                "at": now_iso(),
            },
            last_error_summary=None,
        )
        write_status(status_json_path, state)
    else:
        log("Running Codex preflight.", loop_log)
        preflight = run_codex_preflight(repo_root=repo_root, env=codex_env, model=args.codex_model)
        if not preflight["success"]:
            log(f"CODEX_PREFLIGHT_FAIL {preflight['summary']}", loop_log)
            update_runtime_state(
                state,
                repo_root,
                args.results_tsv,
                loop_state="blocked",
                last_preflight={
                    "status": "failed",
                    "summary": preflight["summary"],
                    "exit_code": preflight["exit_code"],
                    "at": now_iso(),
                },
                last_error_summary=preflight["summary"],
            )
            write_status(status_json_path, state)
            return 1

        log("CODEX_PREFLIGHT_OK", loop_log)
        update_runtime_state(
            state,
            repo_root,
            args.results_tsv,
            loop_state="ready",
            last_preflight={
                "status": "success",
                "summary": preflight["summary"],
                "exit_code": preflight["exit_code"],
                "at": now_iso(),
            },
            last_error_summary=None,
        )
        write_status(status_json_path, state)

    baseline_ok = ensure_baseline(
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
    update_runtime_state(
        state,
        repo_root,
        args.results_tsv,
        loop_state="running",
        last_decision="baseline_keep" if baseline_ok else "baseline_failed",
        last_error_summary=None if baseline_ok else "baseline experiment failed",
    )
    write_status(status_json_path, state)

    round_idx = 0
    while True:
        if args.max_rounds and round_idx >= args.max_rounds:
            log("Reached max rounds. Stopping.", loop_log)
            update_runtime_state(
                state,
                repo_root,
                args.results_tsv,
                loop_state="stopped",
                last_error_summary=None,
            )
            write_status(status_json_path, state)
            break

        round_idx += 1
        start_head = current_head(repo_root)
        log(f"Round {round_idx}: starting from {start_head[:7]}", loop_log)
        update_runtime_state(
            state,
            repo_root,
            args.results_tsv,
            loop_state="running",
            round_idx=round_idx,
            last_decision="running_round",
            last_error_summary=None,
        )
        write_status(status_json_path, state)

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
            summary = f"codex exit={codex_proc.returncode} without new commit"
            log(f"Round {round_idx}: no commit produced ({summary}). Skipping experiment.", loop_log)
            update_runtime_state(
                state,
                repo_root,
                args.results_tsv,
                last_decision="no_commit",
                last_error_summary=summary,
            )
            write_status(status_json_path, state)
            time.sleep(args.sleep_seconds)
            continue

        description = latest_commit_subject(repo_root)
        log(f"Round {round_idx}: testing commit {end_head[:7]} {description}", loop_log)
        update_runtime_state(
            state,
            repo_root,
            args.results_tsv,
            last_decision="testing",
            last_error_summary=None,
        )
        write_status(status_json_path, state)

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
            update_runtime_state(
                state,
                repo_root,
                args.results_tsv,
                last_decision="keep",
                last_error_summary=None,
            )
        else:
            log(f"Round {round_idx}: discard {end_head[:7]} and reset HEAD~1", loop_log)
            sh(["git", "reset", "--hard", "HEAD~1"], cwd=repo_root, check=True, capture_output=False)
            update_runtime_state(
                state,
                repo_root,
                args.results_tsv,
                last_decision="discard",
                last_error_summary=None,
            )

        write_status(status_json_path, state)
        time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
