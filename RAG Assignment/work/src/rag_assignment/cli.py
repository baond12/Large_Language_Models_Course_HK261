from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from .common import read_json, runtime_info, write_json


def doctor(root: Path, require_cuda: bool = False) -> dict:
    info = runtime_info()
    packages = ["numpy", "scipy", "scikit-learn", "openpyxl", "langchain-text-splitters", "torch", "transformers",
                "sentence-transformers", "faiss-cpu", "huggingface-hub", "accelerate"]
    info["packages"] = {}
    for name in packages:
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    try:
        import torch
        info.update(cuda_available=torch.cuda.is_available(), torch_cuda_build=torch.version.cuda)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
            info["capability"] = list(torch.cuda.get_device_capability())
            info["compiled_cuda_architectures"] = torch.cuda.get_arch_list()
            info["gpu_total_memory"] = torch.cuda.get_device_properties(0).total_memory
            # Actual kernel check catches images that detect P100 but lack sm_60 kernels.
            tensor = torch.ones((16, 16), device="cuda", dtype=torch.float16)
            info["cuda_fp16_matmul_test"] = float((tensor @ tensor)[0, 0].item()) == 16.0
    except ImportError:
        info["cuda_available"] = False
    except Exception as error:
        info["cuda_kernel_error"] = f"{type(error).__name__}: {error}"
    write_json(root / "artifacts" / "environment.json", info)
    if require_cuda and (not info.get("cuda_available") or not info.get("cuda_fp16_matmul_test")):
        raise RuntimeError("CUDA/P100 kernel preflight failed; inspect artifacts/environment.json. Use a Pascal-compatible PyTorch CUDA build")
    return info


def lock_models(root: Path) -> dict:
    from huggingface_hub import HfApi
    lock_path = root / "models.lock.json"
    if lock_path.exists():
        return read_json(lock_path)
    api = HfApi()
    result = {}
    for name in ["keepitreal/vietnamese-sbert", "Qwen/Qwen1.5-1.8B"]:
        info = api.model_info(name)
        if not info.sha or len(info.sha) != 40:
            raise ValueError(f"Could not resolve immutable revision for {name}")
        result[name] = {"revision": info.sha, "source": f"https://huggingface.co/{name}"}
    write_json(lock_path, result)
    return result


def main(root: Path | None = None):
    root = root or Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Reproducible Vietnamese law RAG assignment")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor")
    p.add_argument("--require-cuda", action="store_true")
    commands.add_parser("lock-models")
    p = commands.add_parser("prepare")
    p.add_argument("--source-root", type=Path, default=root.parent)
    p.add_argument("--data-dir", type=Path, default=root / "data")
    p.add_argument("--seed", type=int, default=261)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--dev-size", type=int, default=100)
    for command in ["chunks", "bm25", "dense"]:
        p = commands.add_parser(command)
        p.add_argument("--index-dir", type=Path, default=root / "indexes" / "baseline")
        if command == "chunks":
            p.add_argument("--data-dir", type=Path, default=root / "data")
            p.add_argument("--mode", choices=["baseline", "legal"], default="baseline")
        if command == "dense":
            p.add_argument("--device", default="cuda")
            p.add_argument("--batch-size", type=int, default=32)
    for command in ["retrieve", "generate"]:
        p = commands.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--data-dir", type=Path, default=root / "data")
        p.add_argument("--split", choices=["dev", "test", "diagnostic_dev", "diagnostic_test"], default="dev")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cpu" if command == "retrieve" else "auto")
        if command == "generate":
            p.add_argument("--limit", type=int)
    p = commands.add_parser("score")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=root / "data")
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--minimum", type=int, default=100)
    p = commands.add_parser("select")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=root / "artifacts" / "selection.json")
    p = commands.add_parser("demo")
    p.add_argument("--config", type=Path, default=root / "configs" / "r2_grounded.json")
    p.add_argument("--question", required=True)
    p.add_argument("--device", default="auto")
    commands.add_parser("report")
    p = commands.add_parser("package")
    p.add_argument("--kind", choices=["source", "results"], default="results")
    args = parser.parse_args()
    c = args.command
    if c == "doctor":
        result = doctor(root, args.require_cuda)
    elif c == "lock-models":
        result = lock_models(root)
    elif c == "prepare":
        from .data import prepare
        result = prepare(args.source_root.resolve(), args.data_dir.resolve(), args.seed, args.test_size, args.dev_size)
    elif c == "chunks":
        from .chunks import build_chunks
        result = build_chunks(args.data_dir.resolve(), args.index_dir.resolve(), args.mode)
    elif c == "bm25":
        from .chunks import ChunkStore
        from .retrieval import BM25Index
        store = ChunkStore(args.index_dir)
        result = BM25Index.build(store, args.index_dir / "bm25")
        store.close()
    elif c == "dense":
        from .chunks import ChunkStore
        from .retrieval import build_dense
        lock = read_json(root / "models.lock.json")
        model = "keepitreal/vietnamese-sbert"
        store = ChunkStore(args.index_dir)
        result = build_dense(store, args.index_dir / "dense", model, args.device, args.batch_size, lock[model]["revision"])
        store.close()
    elif c == "retrieve":
        from .experiment import retrieval_evaluation
        result = retrieval_evaluation(read_json(args.config), root, args.data_dir, args.split, args.output, args.device)
    elif c == "generate":
        from .experiment import run_generation
        result = run_generation(read_json(args.config), root, args.data_dir, args.split, args.output, args.device, args.limit)
    elif c == "score":
        from .experiment import score_run
        result = score_run(args.run_dir, args.data_dir, root, args.device)
    elif c == "compare":
        from .experiment import compare_runs
        result = compare_runs(args.baseline, args.candidate, args.output, args.minimum, args.split)
    elif c == "select":
        from .experiment import select_candidate
        result = select_candidate(args.runs, args.output)
    elif c == "demo":
        import torch
        from .experiment import make_retriever
        from .generation import LocalGenerator
        config = read_json(args.config)
        lock = read_json(root / "models.lock.json")
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        retriever = make_retriever(config, root, device)
        generator = LocalGenerator(config["llm"], lock[config["llm"]]["revision"], device, config["max_input_tokens"], config["max_new_tokens"])
        hits = retriever.search(args.question, config.get("k", 5))
        result = {"question": args.question, "retrieved": hits,
                  **generator.generate(args.question, hits, config["prompt"], config["packing"])}
        retriever.store.close()
        write_json(root / "artifacts" / "last_demo.json", result)
    elif c == "report":
        from .reporting import build_report
        result = build_report(root)
    elif c == "package":
        from .reporting import make_package
        result = make_package(root, args.kind)
    else:
        raise AssertionError(c)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
