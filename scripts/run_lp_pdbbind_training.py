"""Train three-seed BioLM-Score and GenScore models on LP-PDBBind.

This coordinator deliberately changes no model/training mathematics.  It only
adds bounded GPU scheduling, checkpoint validation, retries, atomic status
files, and resource guards around ``scripts/train_lp_pdbbind.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    model_type: str
    seed: int

    @property
    def name(self) -> str:
        prefix = "biolm_score" if self.model_type == "biolm" else "genscore"
        return f"{prefix}_seed{self.seed}"


@dataclass
class RunningJob:
    task: Task
    gpu: int
    attempt: int
    process: subprocess.Popen
    log_file: object
    checkpoint: Path
    started: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protein-embeddings", type=Path, required=True)
    parser.add_argument("--ligand-embeddings", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--ramp-seconds", type=int, default=180)
    parser.add_argument("--cpu-launch-threshold", type=float, default=80.0)
    parser.add_argument("--min-free-memory-mib", type=int, default=9000)
    parser.add_argument("--max-total-compute-processes-per-gpu", type=int, default=2)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "biolm_score_gpu_locks",
        help="Directory for atomic coordinator GPU lock files.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_fingerprint(train_path: Path, val_path: Path) -> str:
    def values(path: Path) -> list[str]:
        return [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = {
        "train": values(train_path),
        "val": values(val_path),
        "excluded": [],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_cpu_totals() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    return sum(values), idle


def cpu_busy_percent(interval: float = 0.5) -> float:
    total1, idle1 = read_cpu_totals()
    time.sleep(interval)
    total2, idle2 = read_cpu_totals()
    delta_total = max(total2 - total1, 1)
    return 100.0 * (1.0 - (idle2 - idle1) / delta_total)


def gpu_snapshot(gpu: int) -> dict:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if gpu_result.returncode:
        return {"gpu": gpu, "available": False, "error": gpu_result.stderr.strip()}
    memory_used, memory_free, memory_total, utilization = (
        int(part.strip()) for part in gpu_result.stdout.strip().split(",", 3)
    )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    processes = []
    for line in process_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3 and fields[0].isdigit():
            processes.append(
                {
                    "pid": int(fields[0]),
                    "used_memory_mib": fields[1],
                    "process_name": fields[2],
                }
            )
    return {
        "gpu": gpu,
        "available": True,
        "memory_used_mib": memory_used,
        "memory_free_mib": memory_free,
        "memory_total_mib": memory_total,
        "utilization_percent": utilization,
        "compute_processes": processes,
    }


def checkpoint_path(experiment_dir: Path, task: Task) -> Path:
    return experiment_dir / "checkpoints" / f"{task.name}_joint_gatedgcn_alpha1.pth"


def checkpoint_complete(
    experiment_dir: Path, task: Task, expected_split_fingerprint: str
) -> bool:
    checkpoint = checkpoint_path(experiment_dir, task)
    metadata = read_json(checkpoint.with_suffix(".json"))
    train_args = metadata.get("args", {})
    split_audit = metadata.get("split_audit", {})
    numeric_checks = {
        "paper_alpha": (metadata.get("paper_alpha"), 1.0),
        "lr": (train_args.get("lr"), 1e-3),
        "weight_decay": (train_args.get("weight_decay"), 1e-5),
        "aux_weight": (train_args.get("aux_weight"), 1e-3),
        "batch_size": (train_args.get("batch_size"), 64),
    }
    return (
        checkpoint.is_file()
        and checkpoint.stat().st_size > 0
        and metadata.get("stage") == "joint"
        and metadata.get("model_type") == task.model_type
        and metadata.get("ablation") == "full"
        and metadata.get("seed") == task.seed
        and train_args.get("encoder") == "gatedgcn"
        and split_audit.get("mode") == "explicit_ids"
        and split_audit.get("train_final_count") == 7393
        and split_audit.get("val_final_count") == 958
        and split_audit.get("excluded_overlap_after_filter_count") == 0
        and split_audit.get("split_fingerprint") == expected_split_fingerprint
        and all(
            isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12
            for actual, expected in numeric_checks.values()
        )
    )


def build_command(
    args: argparse.Namespace,
    task: Task,
    train_ids: Path,
    val_ids: Path,
) -> list[str]:
    checkpoint = checkpoint_path(args.experiment_dir, task)
    tensorboard = args.experiment_dir / "tensorboard" / task.name
    return [
        str(args.python),
        str(args.trainer),
        "--model-type",
        task.model_type,
        "--ablation",
        "full",
        "--stage",
        "joint",
        "--seed",
        str(task.seed),
        "--output",
        str(checkpoint),
        "--tensorboard-dir",
        str(tensorboard),
        "--alpha",
        "1.0",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.00001",
        "--aux-weight",
        "0.001",
        "--batch-size",
        "64",
        "--num-epochs",
        "2000",
        "--patience",
        "70",
        "--num-workers",
        str(args.num_workers),
        "--encoder",
        "gatedgcn",
        "--hidden-dim",
        "128",
        "--n-gaussians",
        "10",
        "--dropout",
        "0.15",
        "--train-distance-cutoff",
        "7.0",
        "--score-distance-cutoff",
        "5.0",
        "--data-dir",
        str(args.data_dir),
        "--protein-embeddings",
        str(args.protein_embeddings),
        "--ligand-embeddings",
        str(args.ligand_embeddings),
        "--train-ids-file",
        str(train_ids),
        "--val-ids-file",
        str(val_ids),
    ]


def acquire_gpu_locks(
    experiment_dir: Path, gpus: list[int], lock_dir: Path
) -> dict[int, Path]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    locks: dict[int, Path] = {}
    for gpu in gpus:
        path = lock_dir / f"biolm_score_gpu_{gpu}.lock"
        payload = {
            "coordinator_pid": os.getpid(),
            "gpu": gpu,
            "slots": 1,
            "task": experiment_dir.name,
            "experiment_dir": str(experiment_dir),
            "created": time.time(),
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(f"GPU {gpu} lock already exists: {path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
        locks[gpu] = path
    return locks


def release_owned_locks(locks: dict[int, Path]) -> None:
    for path in locks.values():
        metadata = read_json(path)
        if metadata.get("coordinator_pid") == os.getpid():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    args = parse_args()
    args.python = args.python.resolve()
    args.trainer = args.trainer.resolve()
    args.repo_root = args.repo_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    args.split_dir = args.split_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.protein_embeddings = args.protein_embeddings.resolve()
    args.ligand_embeddings = args.ligand_embeddings.resolve()
    args.lock_dir = args.lock_dir.resolve()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("all seeds must be unique")
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError("all GPU indices must be unique")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be positive")

    train_ids = args.split_dir / "train_ids.txt"
    val_ids = args.split_dir / "val_ids.txt"
    for path in (
        args.python,
        args.trainer,
        args.data_dir / "v2020_train_ids.npy",
        args.data_dir / "v2020_train_lig.pt",
        args.data_dir / "v2020_train_prot.pt",
        args.protein_embeddings,
        args.ligand_embeddings,
        train_ids,
        val_ids,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (args.experiment_dir / "logs").mkdir(exist_ok=True)
    (args.experiment_dir / "tensorboard").mkdir(exist_ok=True)
    status_path = args.experiment_dir / "status.json"
    expected_split_fingerprint = split_fingerprint(train_ids, val_ids)
    tasks = [
        Task(model_type, seed)
        for model_type in ("biolm", "genscore")
        for seed in args.seeds
    ]

    manifest = {
        "created": time.time(),
        "experiment_dir": str(args.experiment_dir),
        "repo_root": str(args.repo_root),
        "python": str(args.python),
        "trainer": str(args.trainer),
        "trainer_sha256": sha256(args.trainer),
        "coordinator_sha256": sha256(Path(__file__).resolve()),
        "seeds": args.seeds,
        "models": ["biolm", "genscore"],
        "gpus": args.gpus,
        "slots_per_gpu": args.slots_per_gpu,
        "data_dir": str(args.data_dir),
        "protein_embeddings": str(args.protein_embeddings),
        "ligand_embeddings": str(args.ligand_embeddings),
        "split_fingerprint": expected_split_fingerprint,
        "split_files": {
            str(path): {"sha256": sha256(path), "line_count": len(path.read_text().splitlines())}
            for path in (train_ids, val_ids)
        },
        "training": {
            "stage": "joint",
            "encoder": "gatedgcn",
            "alpha": 1.0,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "aux_weight": 1e-3,
            "batch_size": 64,
            "num_epochs": 2000,
            "patience": 70,
            "dropout": 0.15,
            "train_distance_cutoff": 7.0,
            "score_distance_cutoff": 5.0,
            "num_workers": args.num_workers,
        },
        "resource_guards": {
            "cpu_launch_threshold": args.cpu_launch_threshold,
            "min_free_memory_mib_for_new_job": args.min_free_memory_mib,
            "max_total_compute_processes_per_gpu": (
                args.max_total_compute_processes_per_gpu
            ),
            "ramp_seconds": args.ramp_seconds,
        },
    }
    atomic_json(args.experiment_dir / "manifest.json", manifest)

    attempts = {task.name: 0 for task in tasks}
    cooldown_until: dict[str, float] = {}
    exhausted: dict[str, dict] = {}
    running: dict[str, RunningJob] = {}
    history: list[dict] = []
    last_launch = {gpu: 0.0 for gpu in args.gpus}
    locks = acquire_gpu_locks(args.experiment_dir, args.gpus, args.lock_dir)

    try:
        while True:
            now = time.time()
            for name, job in list(running.items()):
                returncode = job.process.poll()
                if returncode is None:
                    continue
                job.log_file.close()
                valid = returncode == 0 and checkpoint_complete(
                    args.experiment_dir, job.task, expected_split_fingerprint
                )
                event = {
                    "event": "completed" if valid else "failed",
                    "name": name,
                    "gpu": job.gpu,
                    "attempt": job.attempt,
                    "pid": job.process.pid,
                    "returncode": returncode,
                    "valid_checkpoint": valid,
                    "elapsed_seconds": now - job.started,
                    "finished": now,
                }
                history.append(event)
                del running[name]
                if not valid:
                    if job.attempt >= args.max_attempts:
                        exhausted[name] = event
                    else:
                        cooldown_until[name] = now + args.retry_cooldown_seconds

            completed = {
                task.name
                for task in tasks
                if checkpoint_complete(
                    args.experiment_dir, task, expected_split_fingerprint
                )
            }
            pending = [
                task
                for task in tasks
                if task.name not in completed
                and task.name not in running
                and task.name not in exhausted
                and now >= cooldown_until.get(task.name, 0.0)
            ]
            snapshots = {gpu: gpu_snapshot(gpu) for gpu in args.gpus}
            owned_pids = {job.process.pid for job in running.values()}
            assigned_counts = {
                gpu: sum(job.gpu == gpu for job in running.values()) for gpu in args.gpus
            }
            cpu_percent = cpu_busy_percent()

            # Add at most one new process per GPU in a polling cycle.  This
            # launches one process on all six GPUs first, observes real memory,
            # and only then fills the second slot.
            eligible: list[int] = []
            if cpu_percent < args.cpu_launch_threshold:
                for gpu in args.gpus:
                    snapshot = snapshots[gpu]
                    foreign_processes = [
                        process
                        for process in snapshot.get("compute_processes", ())
                        if process["pid"] not in owned_pids
                    ]
                    if (
                        snapshot.get("available")
                        and len(snapshot.get("compute_processes", ()))
                        < args.max_total_compute_processes_per_gpu
                        and assigned_counts[gpu] < args.slots_per_gpu
                        and snapshot.get("memory_free_mib", 0) >= args.min_free_memory_mib
                        and now - last_launch[gpu] >= args.ramp_seconds
                    ):
                        eligible.append(gpu)

            for gpu, task in zip(eligible, pending):
                checkpoint = checkpoint_path(args.experiment_dir, task)
                task_log = args.experiment_dir / "logs" / f"{task.name}_joint.log"
                attempts[task.name] += 1
                attempt = attempts[task.name]
                command = build_command(args, task, train_ids, val_ids)
                log_file = task_log.open("a", encoding="utf-8")
                log_file.write(
                    json.dumps(
                        {
                            "event": "launch",
                            "attempt": attempt,
                            "gpu": gpu,
                            "command": command,
                            "time": now,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log_file.flush()
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONUNBUFFERED": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                    }
                )
                process = subprocess.Popen(
                    command,
                    cwd=args.repo_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                running[task.name] = RunningJob(
                    task=task,
                    gpu=gpu,
                    attempt=attempt,
                    process=process,
                    log_file=log_file,
                    checkpoint=checkpoint,
                    started=now,
                )
                last_launch[gpu] = now
                history.append(
                    {
                        "event": "launched",
                        "name": task.name,
                        "gpu": gpu,
                        "attempt": attempt,
                        "pid": process.pid,
                        "command": shlex.join(command),
                        "launched": now,
                    }
                )

            completed = {
                task.name
                for task in tasks
                if checkpoint_complete(
                    args.experiment_dir, task, expected_split_fingerprint
                )
            }
            waiting = [
                task.name
                for task in tasks
                if task.name not in completed
                and task.name not in running
                and task.name not in exhausted
            ]
            if len(completed) == len(tasks):
                state = "complete"
            elif exhausted and not running and not waiting:
                state = "failed"
            elif pending and cpu_percent >= args.cpu_launch_threshold:
                state = "waiting_for_cpu"
            else:
                state = "training"
            status = {
                "state": state,
                "total_count": len(tasks),
                "completed_count": len(completed),
                "completed": sorted(completed),
                "active": {
                    name: {
                        "model_type": job.task.model_type,
                        "seed": job.task.seed,
                        "gpu": job.gpu,
                        "attempt": job.attempt,
                        "pid": job.process.pid,
                        "checkpoint": str(job.checkpoint),
                        "started": job.started,
                    }
                    for name, job in running.items()
                },
                "waiting": waiting,
                "exhausted": exhausted,
                "cpu_busy_percent": cpu_percent,
                "gpu_snapshots": snapshots,
                "history": history[-200:],
                "updated": time.time(),
            }
            atomic_json(status_path, status)
            if state == "complete":
                atomic_json(
                    args.experiment_dir / "complete.json",
                    {
                        "completed": sorted(completed),
                        "count": len(completed),
                        "finished": time.time(),
                        "manifest": str(args.experiment_dir / "manifest.json"),
                    },
                )
                return
            if state == "failed":
                raise SystemExit(1)
            time.sleep(args.poll_seconds)
    finally:
        for job in running.values():
            try:
                job.log_file.flush()
            except Exception:
                pass
        release_owned_locks(locks)


if __name__ == "__main__":
    main()
