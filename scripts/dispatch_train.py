#!/usr/bin/env python3
"""Dispatch multi-task multi-algorithm training jobs.

This launcher centralizes experiment scheduling for combinations of:
- algorithm (e.g. crl, ppo)
- environment (e.g. ant, ant_u_maze)
- seed list

Config priority (low -> high):
1) COMMON_DEFAULTS
2) ALGO_DEFAULTS[algo]
3) ENV_OVERRIDES[env]
4) TASK_OVERRIDES[(algo, env)]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Shared run settings (applies to all tasks unless overridden below).
COMMON_DEFAULTS: dict[str, Any] = {
    "total_env_steps": 100_000_000,
    "num_evals": 50,
    "num_envs": 1024,
    "episode_length": 1000,
    "action_repeat": 1,
}


# Algorithm defaults (only include flags valid for that algorithm).
ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "crl": {
        "batch_size": 256,
        "discounting": 0.99,
        "unroll_length": 62,
        "min_replay_size": 1000,
        "max_replay_size": 10000,
        "contrastive_loss_fn": "sym_infonce",
        "energy_fn": "l2",
        "train_step_multiplier": 1,
        "policy_lr": 6e-4,
        "critic_lr": 3e-4,
        "logsumexp_penalty_coeff": 0.1,
        "h_dim": 256,
        "n_hidden": 2,
        "repr_dim": 64,
    },
    "ppo": {
        "batch_size": 64,
        "discounting": 0.99,
        "unroll_length": 10,
        "num_minibatches": 16,
        "num_updates_per_batch": 2,
        "learning_rate": 1e-4,
        "entropy_cost": 1e-4,
    },
    "sac": {
        "batch_size": 256,
        "discounting": 0.99,
        "unroll_length": 62,
        "learning_rate": 1e-4,
        "min_replay_size": 1000,
        "max_replay_size": 10000,
        "train_step_multiplier": 1,
        "h_dim": 256,
        "n_hidden": 2,
    },
}


# Environment-wide overrides, independent of algorithm.
ENV_OVERRIDES: dict[str, dict[str, Any]] = {
    "humanoid": {"num_envs": 512},
    "humanoid_u_maze": {"num_envs": 512},
    "humanoid_big_maze": {"num_envs": 512},
    "humanoid_hardest_maze": {"num_envs": 512},
}


# Fine-grained per (algorithm, env) overrides.
TASK_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    # ("crl", "ant_u_maze"): {
    #     "total_env_steps": 15_000_000,
    # },
    # ("ppo", "ant_u_maze"): {
    #     "total_env_steps": 15_000_000,
    #     "num_evals": 100,
    # },
}


@dataclass
class Job:
    algorithm: str
    env_name: str
    seed: int
    command: list[str]
    base_env_vars: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch batched JaxGCRL training jobs.")
    parser.add_argument("--algorithms", nargs="+", default=["crl"], help="Algorithms to run (e.g. crl ppo).")
    parser.add_argument("--envs", nargs="+", default=["ant"], help="Environment names to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seed list.")
    parser.add_argument("--project-name", default="jaxgcrl", help="W&B project name.")
    parser.add_argument(
        "--group-prefix",
        default="multi_task",
        help="Prefix for W&B group. Final group is <prefix>_<algo>_<env>.",
    )
    parser.add_argument("--python-bin", default="python", help="Python executable.")
    parser.add_argument("--run-file", default="run.py", help="Training entry file.")
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help="GPU IDs used for parallel scheduling, e.g. --gpu-ids 0 1 2",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Deprecated. Comma-separated GPU IDs, e.g. 0,1,2. Prefer --gpu-ids.",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="How many concurrent jobs to run on each GPU.",
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=None,
        help="Optional global cap on concurrent jobs.",
    )
    parser.add_argument("--xla-mem-fraction", default=".95", help="XLA_PYTHON_CLIENT_MEM_FRACTION value.")
    parser.add_argument("--mujoco-gl", default="egl", help="MUJOCO_GL value.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when any job fails.",
    )
    parser.add_argument("--log-wandb", dest="log_wandb", action="store_true", default=True)
    parser.add_argument("--no-log-wandb", dest="log_wandb", action="store_false")
    return parser.parse_args()


def merge_task_flags(algorithm: str, env_name: str) -> dict[str, Any]:
    if algorithm not in ALGO_DEFAULTS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {sorted(ALGO_DEFAULTS)}")

    merged = dict(COMMON_DEFAULTS)
    merged.update(ALGO_DEFAULTS[algorithm])
    merged.update(ENV_OVERRIDES.get(env_name, {}))
    merged.update(TASK_OVERRIDES.get((algorithm, env_name), {}))
    return merged


def append_flag(command: list[str], key: str, value: Any):
    if value is None:
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        else:
            command.append(f"--no-{key}")
        return
    command.extend([flag, str(value)])


def resolve_gpu_ids(args: argparse.Namespace) -> list[str]:
    if args.gpu_ids:
        return [str(x) for x in args.gpu_ids]
    if args.cuda_visible_devices:
        return [x.strip() for x in str(args.cuda_visible_devices).split(",") if x.strip()]
    return ["0"]


def build_gpu_slots(args: argparse.Namespace) -> list[str]:
    if args.jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be >= 1")

    gpu_ids = resolve_gpu_ids(args)
    if not gpu_ids:
        raise ValueError("No GPU IDs resolved. Provide --gpu-ids or --cuda-visible-devices")

    slots: list[str] = []
    for gpu_id in gpu_ids:
        slots.extend([gpu_id] * args.jobs_per_gpu)

    if args.max_parallel_jobs is not None:
        if args.max_parallel_jobs < 1:
            raise ValueError("--max-parallel-jobs must be >= 1")
        slots = slots[: args.max_parallel_jobs]

    if not slots:
        raise ValueError("No available execution slots after applying parallelism limits.")
    return slots


def build_job(args: argparse.Namespace, algorithm: str, env_name: str, seed: int) -> Job:
    exp_name = f"{algorithm}_{env_name}_s{seed}"
    group_name = f"{args.group_prefix}_{algorithm}_{env_name}"
    checkpoint_logdir = f"./runs/run_{exp_name}_s_{seed}/ckpt"

    task_flags = merge_task_flags(algorithm, env_name)
    command = [
        args.python_bin,
        args.run_file,
        algorithm,
        "--env",
        env_name,
        "--seed",
        str(seed),
        "--exp_name",
        exp_name,
        "--wandb_project_name",
        args.project_name,
        "--wandb_group",
        group_name,
        "--checkpoint_logdir",
        checkpoint_logdir,
    ]
    if args.log_wandb:
        command.append("--log_wandb")
    else:
        command.append("--no-log_wandb")

    for k, v in task_flags.items():
        append_flag(command, k, v)

    env_vars = os.environ.copy()
    env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction
    env_vars["MUJOCO_GL"] = args.mujoco_gl

    return Job(algorithm=algorithm, env_name=env_name, seed=seed, command=command, base_env_vars=env_vars)


def command_to_str(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main():
    args = parse_args()
    run_file = Path(args.run_file)
    if not run_file.exists():
        raise FileNotFoundError(f"Cannot find run file: {run_file.resolve()}")

    jobs: list[Job] = []
    for algorithm in args.algorithms:
        for env_name in args.envs:
            for seed in args.seeds:
                jobs.append(build_job(args, algorithm, env_name, seed))

    slots = build_gpu_slots(args)

    print(f"Prepared {len(jobs)} job(s).")
    print(f"Scheduling slots: {slots}")
    for i, job in enumerate(jobs, 1):
        preview_gpu = slots[(i - 1) % len(slots)]
        print(
            f"[{i}/{len(jobs)}] gpu={preview_gpu} algo={job.algorithm} env={job.env_name} seed={job.seed}\n"
            f"  {command_to_str(job.command)}"
        )

    if args.dry_run:
        print("\nDry run only, no command executed.")
        return

    pending = list(jobs)
    active: list[tuple[subprocess.Popen, Job, str]] = []
    available_slots = list(slots)
    failures: list[tuple[Job, int]] = []
    completed = 0

    while pending or active:
        while pending and available_slots:
            job = pending.pop(0)
            slot_gpu = available_slots.pop(0)
            env_vars = job.base_env_vars.copy()
            env_vars["CUDA_VISIBLE_DEVICES"] = slot_gpu
            print(
                f"\n>>> Running [{completed + len(active) + 1}/{len(jobs)}] gpu={slot_gpu} "
                f"{job.algorithm} {job.env_name} seed={job.seed}"
            )
            proc = subprocess.Popen(job.command, env=env_vars)
            active.append((proc, job, slot_gpu))

        if not active:
            continue

        time.sleep(1.0)
        next_active: list[tuple[subprocess.Popen, Job, str]] = []
        for proc, job, slot_gpu in active:
            return_code = proc.poll()
            if return_code is None:
                next_active.append((proc, job, slot_gpu))
                continue

            completed += 1
            available_slots.append(slot_gpu)
            if return_code != 0:
                failures.append((job, return_code))
                print(
                    f"!!! Failed [{completed}/{len(jobs)}] gpu={slot_gpu} "
                    f"{job.algorithm} {job.env_name} seed={job.seed} exit={return_code}"
                )
            else:
                print(
                    f">>> Finished [{completed}/{len(jobs)}] gpu={slot_gpu} "
                    f"{job.algorithm} {job.env_name} seed={job.seed}"
                )

        active = next_active
        if args.fail_fast and failures:
            print("\nFail-fast enabled. Terminating remaining running jobs...")
            for proc, _, _ in active:
                terminate_process(proc)
            pending.clear()
            active.clear()
            break

    if failures:
        print("\nFailed jobs:")
        for job, code in failures:
            print(f"- algo={job.algorithm} env={job.env_name} seed={job.seed} exit={code}")
        raise SystemExit(1)

    print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()
