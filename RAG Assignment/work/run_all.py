"""Run in separate processes so CUDA memory is released between experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command(*args):
    print("\nRUN", " ".join(str(x) for x in args), flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "run.py"), *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT.parent)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stage", choices=["prepare", "pilot", "full"], default="full")
    parser.add_argument("--include-legal", action="store_true")
    parser.add_argument("--include-instruct", action="store_true", help="Also evaluate the controlled instruct/chat ablations")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.stage == "prepare":
        command("doctor")
    else:
        command("doctor", *(["--require-cuda"] if args.device == "cuda" else []))
    command("prepare", "--source-root", args.source_root.resolve())
    subprocess.run([sys.executable, str(ROOT / "scripts/validate.py")], cwd=ROOT, check=True)
    modes = ["baseline", "legal"] if args.include_legal else ["baseline"]
    for mode in modes:
        index = ROOT / "indexes" / mode
        command("chunks", "--mode", mode, "--index-dir", index)
        command("bm25", "--index-dir", index)
    if args.stage == "prepare":
        command("retrieve", "--config", "configs/sparse_diagnostic.json", "--split", "diagnostic_dev",
                "--output", "artifacts/retrieval_dev/sparse_diagnostic", "--device", "cpu")
        command("report")
        return
    command("lock-models", *(["--model", "Qwen/Qwen2.5-3B-Instruct"] if args.include_instruct else []))
    for mode in modes:
        command("dense", "--index-dir", ROOT / "indexes" / mode, "--device", args.device, "--batch-size", args.batch_size)
    names = ["b0_code", "b1_cosine", "r1_hybrid", "r2_grounded"] + (["r3_legal"] if args.include_legal else [])
    if args.include_instruct:
        names += ["i0_dense_chat", "i1_bm25_chat", "i2_bm25_grounded_chat"]
    for name in names:
        command("retrieve", "--config", f"configs/{name}.json", "--split", "diagnostic_dev",
                "--output", f"artifacts/retrieval_dev/{name}", "--device", args.device)
        folder = f"runs/{'pilot' if args.stage == 'pilot' else 'dev'}/{name}"
        command("generate", "--config", f"configs/{name}.json", "--split", "dev", "--output", folder,
                "--device", args.device, *(["--limit", "3"] if args.stage == "pilot" else []))
        command("score", "--run-dir", folder, "--device", args.device)
        summary = json.loads((ROOT / folder / "summary.json").read_text(encoding="utf-8"))
        if args.stage == "pilot" and summary["errors"]:
            raise RuntimeError(f"Pilot {name} has inference errors; inspect {folder}/predictions.jsonl before full run")
        if summary["errors"] == summary["questions"]:
            raise RuntimeError(f"All questions failed in {folder}; fix runtime before continuing")
    if args.stage == "pilot":
        command("report")
        command("package", "--kind", "results")
        return
    command("select", "--runs", *[f"runs/dev/{name}" for name in names])
    selection = json.loads((ROOT / "artifacts/selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_config"]["name"]
    # Primary holdout is evaluated after selection. Diagnostic holdout is not used for tuning.
    for name in dict.fromkeys(["b0_code", "b1_cosine", selected]):
        folder = f"runs/test/{name}"
        command("generate", "--config", f"configs/{name}.json", "--split", "test", "--output", folder, "--device", args.device)
        command("score", "--run-dir", folder, "--device", args.device)
        command("retrieve", "--config", f"configs/{name}.json", "--split", "diagnostic_test",
                "--output", f"artifacts/retrieval_test/{name}", "--device", args.device)
    for baseline in ["b0_code", "b1_cosine"]:
        command("compare", "--baseline", f"runs/test/{baseline}", "--candidate", f"runs/test/{selected}",
                "--output", f"artifacts/comparisons/{baseline}_vs_{selected}")
    command("report")
    command("package", "--kind", "results")


if __name__ == "__main__":
    main()
