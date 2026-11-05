from __future__ import annotations

import json
import time
from pathlib import Path

from .chunks import ChunkStore
from .common import fingerprint, implementation_identity, read_json, read_jsonl, runtime_info, sha256_file, write_json, write_jsonl
from .metrics import cosine_from_vectors, lexical_metrics, paired_bootstrap, retrieval_metrics
from .retrieval import BM25Index, DenseIndex, Retriever, load_encoder


def load_questions(data_dir: Path, split: str) -> list[dict]:
    ids = read_json(data_dir / "splits.json")["ids"][split]
    questions = {q["id"]: q for q in read_jsonl(data_dir / "questions.jsonl")}
    return [questions[i] for i in ids]


def make_retriever(config: dict, root: Path, device: str = "cpu") -> Retriever:
    chunk_dir = root / config["index_dir"]
    store = ChunkStore(chunk_dir)
    bm25 = BM25Index(chunk_dir / "bm25") if config["retriever"] in {"bm25", "hybrid"} else None
    dense = None
    if config["retriever"] in {"dense", "hybrid"}:
        manifest = read_json(chunk_dir / "dense" / "dense_manifest.json")
        encoder = load_encoder(manifest["model"], device, manifest["actual_revision"])
        dense = DenseIndex(chunk_dir / "dense", encoder, config.get("distance", "cosine"))
    return Retriever(store, config["retriever"], bm25, dense, config.get("candidate_k", 30), config.get("rrf_constant", 60))


def retrieval_evaluation(config: dict, root: Path, data_dir: Path, split: str, output: Path, device: str = "cpu") -> dict:
    import numpy as np
    questions = load_questions(data_dir, split)
    retriever = make_retriever(config, root, device)
    records = []
    for q in questions:
        start = time.perf_counter()
        hits = retriever.search(q["question"], config.get("k", 5))
        records.append({**q, "retrieved": hits, "retrieval_seconds": time.perf_counter() - start})
    # Labels are read only after retrieval is complete; never passed to the retriever.
    labels = {r["id"]: r for r in read_jsonl(data_dir / "labels.jsonl")}
    corpus_ids = {r[0] for r in retriever.store.conn.execute("SELECT DISTINCT law_id FROM chunks")}
    covered = []
    for result in records:
        relevant = labels[result["id"]]["law_ids"]
        result["relevant_law_ids"] = relevant
        result["reference_in_corpus"] = bool(set(relevant) & corpus_ids)
        result["metrics"] = retrieval_metrics([h["law_id"] for h in result["retrieved"]], relevant, config.get("k", 5))
        if result["reference_in_corpus"]:
            covered.append(result)
    keys = list(records[0]["metrics"]) if records else []
    summary = {"config": config, "split": split, "questions": len(records), "reference_covered_questions": len(covered),
        "all_questions_metrics": {k: float(np.mean([r["metrics"][k] for r in records])) for k in keys},
        "covered_questions_metrics": {k: float(np.mean([r["metrics"][k] for r in covered])) if covered else None for k in keys},
        "latency_seconds": {"median": float(np.median([r["retrieval_seconds"] for r in records])) if records else None,
                            "p95": float(np.quantile([r["retrieval_seconds"] for r in records], .95)) if records else None},
        "data_fingerprint": read_json(data_dir / "manifest.json")["fingerprint"],
        "chunk_fingerprint": retriever.store.manifest["fingerprint"], "runtime": runtime_info(),
        "limitation": "Weak filename-level labels, not passage annotations. Uncovered references cannot be retrieved as standalone files. Zero denominator yields null, not a retrieval failure estimate."}
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "retrieval.jsonl", records)
    write_json(output / "summary.json", summary)
    retriever.store.close()
    return summary


def run_generation(config: dict, root: Path, data_dir: Path, split: str, output: Path,
                   device: str = "auto", limit: int | None = None) -> dict:
    from .generation import LocalGenerator
    import psutil
    import torch
    from tqdm import tqdm
    questions = load_questions(data_dir, split)
    if limit is not None:
        if limit <= 0:
            raise ValueError("Limit must be positive")
        questions = questions[:limit]
    questions_identity = fingerprint(questions)
    lock = read_json(root / "models.lock.json")
    llm_revision = lock[config["llm"]]["revision"]
    chunk_manifest = read_json(root / config["index_dir"] / "chunks_manifest.json")
    index_identity = {"chunks": chunk_manifest["fingerprint"]}
    for part in ["bm25", "dense"]:
        if part == "bm25" and config["retriever"] in {"bm25", "hybrid"} or part == "dense" and config["retriever"] in {"dense", "hybrid"}:
            manifest = read_json(root / config["index_dir"] / part / f"{part}_manifest.json")
            index_identity[part] = manifest["fingerprint"]
            if part == "dense":
                index_identity["dense_revision"] = manifest["actual_revision"]
    implementation = implementation_identity(root)
    identity = fingerprint({"config": config, "data": read_json(data_dir / "manifest.json"),
        "questions": questions_identity, "llm_revision": llm_revision, "indexes": index_identity,
        "implementation": implementation})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, predictions_path = output / "manifest.json", output / "predictions.jsonl"
    if manifest_path.exists():
        if read_json(manifest_path)["fingerprint"] != identity:
            raise ValueError("Run folder belongs to a different experiment; use a new folder")
    else:
        if predictions_path.exists():
            raise ValueError("Predictions without manifest; refusing to mix runs")
        write_json(manifest_path, {"fingerprint": identity, "config": config, "split": split,
            "question_ids": [q["id"] for q in questions], "data_fingerprint": read_json(data_dir / "manifest.json")["fingerprint"],
            "llm_revision": llm_revision, "index_identity": index_identity, "implementation": implementation, "runtime": runtime_info(),
            "status": "started", "limited_run": limit is not None})
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    seen = {r["id"] for r in existing}
    if len(seen) != len(existing) or not seen.issubset({q["id"] for q in questions}):
        raise ValueError("Duplicate or foreign prediction IDs")
    if len(seen) == len(questions):
        return {"status": "already_complete", "questions": len(questions), "output": str(output)}
    retrieval_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    retriever = make_retriever(config, root, retrieval_device)
    generator = LocalGenerator(config["llm"], llm_revision, device, config.get("max_input_tokens", 4096),
                               config.get("max_new_tokens", 256), config.get("seed", 261),
                               chat_template=config.get("chat_template", False))
    manifest = read_json(manifest_path)
    manifest["generator"] = generator.info
    write_json(manifest_path, manifest)
    process = psutil.Process()
    with predictions_path.open("a", encoding="utf-8") as f:
        for q in tqdm(questions, desc=f"{config['name']} {split}"):
            if q["id"] in seen:
                continue
            result = {**q, "answer": "", "error": None, "retrieved": []}
            start = time.perf_counter()
            try:
                t = time.perf_counter()
                hits = retriever.search(q["question"], config.get("k", 5))
                result["retrieval_seconds"] = time.perf_counter() - t
                result["retrieved"] = hits
                result.update(generator.generate(q["question"], hits, config.get("prompt", "baseline"), config.get("packing", "prefix")))
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result["total_seconds"] = time.perf_counter() - start
            result["process_rss_bytes"] = process.memory_info().rss
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
    retriever.store.close()
    manifest["status"] = "inference_complete"
    manifest["completed_questions"] = len(read_jsonl(predictions_path))
    write_json(manifest_path, manifest)
    return {"status": "inference_complete", "questions": len(questions), "output": str(output)}


def score_run(run_dir: Path, data_dir: Path, root: Path, device: str = "cpu") -> dict:
    import numpy as np
    evaluator_name = "keepitreal/vietnamese-sbert"
    lock = read_json(root / "models.lock.json")
    revision = lock[evaluator_name]["revision"]
    manifest = read_json(run_dir / "manifest.json")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    ids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest["question_ids"]):
        raise ValueError("Run is incomplete or contains duplicate/foreign IDs")
    if manifest["data_fingerprint"] != read_json(data_dir / "manifest.json")["fingerprint"]:
        raise ValueError("Labels/data differ from inference dataset")
    labels = {r["id"]: r for r in read_jsonl(data_dir / "labels.jsonl")}
    encoder = load_encoder(evaluator_name, device, revision)
    records = []
    # Empty/failed answers score zero for all metrics, matching original baseline policy.
    for p in predictions:
        answer = p["answer"] if not p.get("error") else ""
        reference = labels[p["id"]]["answer"]
        metrics = lexical_metrics(answer, reference)
        if answer and reference:
            vectors = encoder.encode([answer, reference], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
            metrics["Cosine_Similarity"] = cosine_from_vectors(vectors[0], vectors[1])
        else:
            metrics["Cosine_Similarity"] = 0.0
        records.append({"id": p["id"], "metrics": metrics, "error": p.get("error"),
                        "reference": reference, "law_ids": labels[p["id"]]["law_ids"]})
    key_metrics = ["Cosine_Similarity", "Jaccard_Similarity", "Token_Overlap", "BLEU_Score", "ROUGE_L",
                   "Token_Overlap_Set_PDF", "BLEU_Set_PDF"]
    summary = {"name": manifest["config"]["name"], "split": manifest["split"], "questions": len(records),
        "generator": manifest.get("generator", {}),
        "errors": sum(bool(p.get("error")) for p in predictions),
        "empty_answers": sum(not p["answer"].strip() for p in predictions),
        "generation_cap_reached": sum(p.get("reached_generation_cap", False) for p in predictions),
        "metrics": {key: float(np.mean([r["metrics"][key] for r in records])) for key in key_metrics},
        "latency_seconds": {"median": float(np.median([p["total_seconds"] for p in predictions])),
                            "p95": float(np.quantile([p["total_seconds"] for p in predictions], .95))},
        "max_observed_process_rss_bytes": max(p.get("process_rss_bytes", 0) for p in predictions),
        "max_observed_gpu_allocated_bytes": max((p.get("gpu_peak_allocated_bytes") or 0) for p in predictions),
        "evaluator": {"embedding": evaluator_name, "revision": revision, "lexical": "baseline_v1_plus_pdf_set_variants",
            "cosine_limit": encoder.max_seq_length,
            "cosine_limitation": "Encoder truncates long answers at its max sequence length, as in baseline; cosine alone does not measure full-answer correctness."},
        "run_fingerprint": manifest["fingerprint"], "data_fingerprint": manifest["data_fingerprint"],
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"), "limited_run": manifest["limited_run"]}
    write_jsonl(run_dir / "scores.jsonl", records)
    summary["scores_sha256"] = sha256_file(run_dir / "scores.jsonl")
    write_json(run_dir / "summary.json", summary)
    return summary


def compare_runs(baseline_dir: Path, candidate_dir: Path, output_dir: Path, minimum: int = 100,
                 required_split: str = "test") -> dict:
    a, b = read_json(baseline_dir / "summary.json"), read_json(candidate_dir / "summary.json")
    if a["split"] != required_split or b["split"] != required_split:
        raise ValueError("Comparison split differs from requested split")
    if a["evaluator"] != b["evaluator"] or a["data_fingerprint"] != b["data_fingerprint"]:
        raise ValueError("Incomparable evaluator or dataset")
    for directory, summary in [(baseline_dir, a), (candidate_dir, b)]:
        if sha256_file(directory / "predictions.jsonl") != summary["predictions_sha256"]:
            raise ValueError("Predictions changed after scoring; rescore")
        if sha256_file(directory / "scores.jsonl") != summary["scores_sha256"]:
            raise ValueError("Scores changed after scoring; rescore")
    x = {r["id"]: r for r in read_jsonl(baseline_dir / "scores.jsonl")}
    y = {r["id"]: r for r in read_jsonl(candidate_dir / "scores.jsonl")}
    if set(x) != set(y) or len(x) < minimum:
        raise ValueError(f"Need same >= {minimum} IDs in both runs")
    comparison = {"baseline": a, "candidate": b, "paired_questions": len(x), "split": required_split,
        "deltas": {key: paired_bootstrap([y[i]["metrics"][key] - x[i]["metrics"][key] for i in sorted(x)]) for key in a["metrics"]},
        "meets_sample_requirement": len(x) >= 100, "shows_any_primary_metric_gain": any(b["metrics"][k] > a["metrics"][k] for k in ["Cosine_Similarity", "Jaccard_Similarity", "Token_Overlap", "BLEU_Score", "ROUGE_L"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", comparison)
    # Predeclared selection by per-question delta: show gains and regressions, not only wins.
    ranked = sorted(x, key=lambda i: y[i]["metrics"]["ROUGE_L"] - x[i]["metrics"]["ROUGE_L"])
    cases = [{"id": i, "rouge_l_delta": y[i]["metrics"]["ROUGE_L"] - x[i]["metrics"]["ROUGE_L"]} for i in ranked[:5] + ranked[-5:]]
    write_json(output_dir / "error_analysis_cases.json", cases)
    return comparison


def select_candidate(run_dirs: list[Path], output: Path) -> dict:
    summaries = [(d, read_json(d / "summary.json")) for d in run_dirs]
    if not summaries or any(s["split"] != "dev" or s["limited_run"] for _, s in summaries):
        raise ValueError("Select only complete un-limited dev runs")
    ids = [set(r["id"] for r in read_jsonl(d / "scores.jsonl")) for d, _ in summaries]
    if any(x != ids[0] for x in ids):
        raise ValueError("Candidate dev question sets differ")
    if any(s["evaluator"] != summaries[0][1]["evaluator"] or s["data_fingerprint"] != summaries[0][1]["data_fingerprint"] for _, s in summaries):
        raise ValueError("Candidate dev evaluators/data differ")
    if all(s["errors"] == s["questions"] for _, s in summaries):
        raise ValueError("All candidate runs failed inference; inspect errors rather than selecting an all-zero winner")
    chosen_dir, chosen = max(summaries, key=lambda pair: (pair[1]["metrics"]["ROUGE_L"], -pair[1]["latency_seconds"]["median"], pair[1]["name"]))
    manifest = read_json(chosen_dir / "manifest.json")
    selection = {"criterion": "max dev ROUGE_L, tie: lower median latency, then name", "selected_config": manifest["config"],
                 "selected_run_fingerprint": chosen["run_fingerprint"], "dev_ids_hash": fingerprint(sorted(ids[0])),
                 "candidates": [{"name": s["name"], "ROUGE_L": s["metrics"]["ROUGE_L"]} for _, s in summaries]}
    write_json(output, selection)
    return selection
