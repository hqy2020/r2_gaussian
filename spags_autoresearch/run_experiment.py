#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

try:
    import yaml
except Exception:
    yaml = None


def run_and_log(cmd, cwd, log_path, append=False):
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc


def query_gpu_mem_mib():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.splitlines():
            line = line.strip()
            if line:
                return int(line)
    except Exception:
        return None
    return None


def monitor_gpu_mem(stop_event, interval_s, max_mem_ref):
    while not stop_event.is_set():
        mem = query_gpu_mem_mib()
        if mem is not None and mem > max_mem_ref[0]:
            max_mem_ref[0] = mem
        stop_event.wait(interval_s)


def find_latest_eval(model_path):
    matches = []
    for root, _, files in os.walk(model_path):
        if "eval2d_render_test.yml" in files:
            path = os.path.join(root, "eval2d_render_test.yml")
            match = re.search(r"iter_(\d+)", path)
            iter_num = int(match.group(1)) if match else -1
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            matches.append((iter_num, mtime, path))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def parse_eval(path):
    if not path or not os.path.exists(path):
        return None, None
    if yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return float(data.get("psnr_2d")), float(data.get("ssim_2d"))
        except Exception:
            pass

    psnr = None
    ssim = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("psnr_2d:"):
                try:
                    psnr = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            if line.startswith("ssim_2d:"):
                try:
                    ssim = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
    return psnr, ssim


def ensure_results_tsv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("commit\tpsnr_2d\tssim_2d\tmemory_gb\tstatus\tdescription\n")


def read_best_keep(path):
    best = None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("commit") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[4].strip() != "keep":
                continue
            try:
                psnr = float(parts[1])
                ssim = float(parts[2])
            except Exception:
                continue
            if best is None or (psnr > best[0]) or (psnr == best[0] and ssim > best[1]):
                best = (psnr, ssim)
    return best


def git_short_rev(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def find_conda_executable():
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        "/root/miniconda3/bin/conda",
        "/root/anaconda3/bin/conda",
        "/opt/conda/bin/conda",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("conda executable not found")


def main():
    parser = argparse.ArgumentParser(description="Run a single SPAGS autoresearch experiment")
    parser.add_argument("--repo-root", default="/root/r2_gaussian")
    parser.add_argument("--dataset", default="/root/r2_gaussian/data/chest_50_3views.pickle")
    parser.add_argument("--model-path", default="/root/r2_gaussian/output/autoresearch_latest")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--time-budget-minutes", type=float, default=15.0)
    parser.add_argument("--conda-env", default="r2gs")
    parser.add_argument("--results-tsv", default="/root/experiments/autoresearch/results.tsv")
    parser.add_argument("--description", default="unspecified")
    parser.add_argument("--extra-train-args", default="")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    repo_root = args.repo_root
    if not os.path.exists(repo_root):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(os.path.dirname(args.results_tsv), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{run_tag}.log")
    conda_exe = find_conda_executable()

    train_cmd = [
        conda_exe,
        "run",
        "-n",
        args.conda_env,
        "python",
        "train.py",
        "-s",
        args.dataset,
        "-m",
        args.model_path,
        "--iterations",
        str(args.iterations),
        "--time_budget_minutes",
        str(args.time_budget_minutes),
    ]
    if args.extra_train_args:
        train_cmd.extend(shlex.split(args.extra_train_args))

    max_mem_ref = [0]
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu_mem,
        args=(stop_event, 5.0, max_mem_ref),
        daemon=True,
    )
    monitor_thread.start()

    proc = run_and_log(train_cmd, cwd=repo_root, log_path=log_path, append=False)
    exit_code = proc.wait()
    stop_event.set()
    monitor_thread.join(timeout=2.0)

    eval_path = find_latest_eval(args.model_path)
    if not eval_path:
        test_cmd = [
            conda_exe,
            "run",
            "-n",
            args.conda_env,
            "python",
            "test.py",
            "-m",
            args.model_path,
        ]
        run_and_log(test_cmd, cwd=repo_root, log_path=log_path, append=True).wait()
        eval_path = find_latest_eval(args.model_path)

    psnr, ssim = parse_eval(eval_path)
    if psnr is None or ssim is None:
        psnr = 0.0
        ssim = 0.0

    mem_gb = max_mem_ref[0] / 1024.0 if max_mem_ref[0] > 0 else 0.0
    status = "crash" if exit_code != 0 else "pending"

    ensure_results_tsv(args.results_tsv)
    best_keep = read_best_keep(args.results_tsv)
    if status != "crash":
        if best_keep is None:
            status = "keep"
        else:
            status = "keep" if (psnr > best_keep[0] and ssim > best_keep[1]) else "discard"

    commit = git_short_rev(repo_root)
    with open(args.results_tsv, "a", encoding="utf-8") as f:
        f.write(
            f"{commit}\t{psnr:.6f}\t{ssim:.6f}\t{mem_gb:.2f}\t{status}\t{args.description}\n"
        )

    best_msg = "none" if best_keep is None else f"psnr={best_keep[0]:.6f} ssim={best_keep[1]:.6f}"
    print(f"RESULT psnr={psnr:.6f} ssim={ssim:.6f} mem_gb={mem_gb:.2f} status={status}")
    print(f"BEST_KEEP {best_msg}")
    print(f"LOG {log_path}")

    return 0 if status == "keep" else 1


if __name__ == "__main__":
    sys.exit(main())
