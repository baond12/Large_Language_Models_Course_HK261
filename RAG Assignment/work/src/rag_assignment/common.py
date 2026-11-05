from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def runtime_info() -> dict:
    info = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        info.update(ram_bytes=psutil.virtual_memory().total, process_rss_bytes=psutil.Process().memory_info().rss)
    except ImportError:
        pass
    return info


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def check_under_4b(parameter_count: int) -> None:
    if not 0 < parameter_count < 4_000_000_000:
        raise ValueError(f"Assignment requires 0 < model parameters < 4B, got {parameter_count}")


def implementation_identity(root: Path) -> dict:
    versions = {}
    for name in ["numpy", "torch", "transformers", "sentence-transformers", "faiss-cpu", "scipy", "scikit-learn"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"source": {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "src").rglob("*.py"))},
            "packages": versions}
