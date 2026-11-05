"""Assignment evaluator: baseline definitions plus explicitly named PDF variants."""
from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("\n", " ").strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def lcs_length(a: list[str], b: list[str]) -> int:
    # Identical LCS result to baseline, O(min(m,n)) rather than O(m*n) memory.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]
        for j, other in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if token == other else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def lexical_metrics(prediction: str, reference: str) -> dict[str, float]:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    ps, rs = set(pred), set(ref)
    intersection = len(ps & rs)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred)) if pred else 0.0
    lcs = lcs_length(pred, ref) if pred and ref else 0
    return {
        "Jaccard_Similarity": intersection / len(ps | rs) if ps and rs else 0.0,
        "Token_Overlap": overlap / len(ref) if ref else 0.0,
        "BLEU_Score": bp * overlap / len(pred) if pred and ref else 0.0,
        "ROUGE_L": 2 * lcs / (len(pred) + len(ref)) if pred and ref else 0.0,
        "Token_Overlap_Set_PDF": intersection / len(rs) if rs else 0.0,
        "BLEU_Set_PDF": bp * intersection / len(ps) if ps and rs else 0.0,
    }


def cosine_from_vectors(a, b) -> float:
    import numpy as np
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)) if denom else 0.0


def retrieval_metrics(retrieved_law_ids: list[str], relevant_law_ids: list[str], k: int = 5) -> dict:
    relevant = set(relevant_law_ids)
    ranked = retrieved_law_ids[:k]
    hits = set(ranked) & relevant
    first = next((i for i, law in enumerate(ranked, 1) if law in relevant), None)
    return {f"Hit@{k}": float(bool(hits)), f"Recall@{k}": len(hits) / len(relevant) if relevant else 0.0,
            f"MRR@{k}": 1.0 / first if first else 0.0}


def paired_bootstrap(differences: list[float], seed: int = 261, samples: int = 2000) -> dict:
    import numpy as np
    if not differences:
        raise ValueError("No paired observations")
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return {"mean_delta": float(values.mean()), "ci95_low": float(np.quantile(estimates, .025)),
            "ci95_high": float(np.quantile(estimates, .975)), "pairs": len(values),
            "bootstrap_samples": samples, "seed": seed,
            "unit": "question; correlations within a legal document are not corrected"}
