import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_gpu_power_watts(gpu_id: int) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def device_from_arg(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


@dataclass
class TimerResult:
    wall_ms: float
    gpu_ms: float | None


class CudaTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"

    def __enter__(self):
        cuda_sync()
        self.t0 = time.perf_counter()
        if self.use_cuda:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e1 = torch.cuda.Event(enable_timing=True)
            self.e0.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda:
            self.e1.record()
        cuda_sync()
        self.t1 = time.perf_counter()
        self.wall_ms = (self.t1 - self.t0) * 1000.0
        self.gpu_ms = self.e0.elapsed_time(self.e1) if self.use_cuda else None

    def result(self) -> TimerResult:
        return TimerResult(wall_ms=self.wall_ms, gpu_ms=self.gpu_ms)


def bytes_to_mb(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0)


def bytes_to_gb(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0 * 1024.0)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    return p
