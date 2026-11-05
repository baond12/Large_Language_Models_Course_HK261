from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

from .chunks import ChunkStore
from .common import fingerprint, read_json, write_json


def top_indices(scores, k: int, positive_only: bool = False) -> list[int]:
    import numpy as np
    scores = np.asarray(scores)
    if scores.size == 0 or k <= 0:
        return []
    valid = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(valid) > k:
        # Stable full sort on the short candidate vector makes tied results reproducible.
        threshold = np.partition(scores[valid], len(valid) - k)[len(valid) - k]
        valid = valid[scores[valid] >= threshold]
    return [int(i) for i in sorted(valid, key=lambda i: (-float(scores[i]), int(i)))[:k]]


def reciprocal_rank_fusion(rankings: list[list[int]], weights: list[float] | None = None,
                           rank_constant: int = 60, k: int = 5) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings) or rank_constant <= 0:
        raise ValueError("Invalid RRF configuration")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] += weight / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class BM25Index:
    """BM25 positive-IDF variant (Lucene style), scipy CSC postings, no pickle."""

    def __init__(self, directory: Path):
        import numpy as np
        from scipy.sparse import load_npz
        from sklearn.feature_extraction.text import CountVectorizer
        self.directory = directory
        self.manifest = read_json(directory / "bm25_manifest.json")
        self.weights = load_npz(directory / "bm25.npz").tocsc()
        vocabulary = read_json(directory / "vocabulary.json")
        self.vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=r"(?u)\b\w+\b", lowercase=True)
        self.analyzer = self.vectorizer.build_analyzer()
        self.vocabulary = vocabulary
        if self.weights.shape[1] != len(vocabulary):
            raise ValueError("BM25 vocabulary/matrix mismatch")

    @staticmethod
    def build(store: ChunkStore, directory: Path, k1: float = 1.5, b: float = .75) -> dict:
        import numpy as np
        from scipy.sparse import save_npz
        from sklearn.feature_extraction.text import CountVectorizer
        params = {"chunk_fingerprint": store.manifest["fingerprint"], "k1": k1, "b": b,
                  "idf": "log(1+(N-df+0.5)/(df+0.5))", "token_pattern": r"(?u)\b\w+\b", "version": 1}
        identity = fingerprint(params)
        manifest_path = directory / "bm25_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["fingerprint"] != identity:
                raise ValueError("BM25 cache configuration changed")
            return manifest
        directory.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True, dtype=np.float32)
        counts = vectorizer.fit_transform(store.texts()).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = float(lengths.mean())
        norm = k1 * (1 - b + b * lengths / max(avgdl, 1e-9))
        counts.data = (counts.data * (k1 + 1)) / (counts.data + np.repeat(norm, np.diff(counts.indptr)))
        weights = counts.tocsc()
        del counts
        df = np.diff(weights.indptr)
        idf = np.log1p((weights.shape[0] - df + .5) / (df + .5)).astype(np.float32)
        weights.data *= np.repeat(idf, df)
        save_npz(directory / "bm25.npz", weights, compressed=False)
        write_json(directory / "vocabulary.json", {term: int(i) for term, i in vectorizer.vocabulary_.items()})
        manifest = {**params, "fingerprint": identity, "documents": weights.shape[0], "terms": weights.shape[1],
                    "nonzeros": weights.nnz, "average_document_tokens": avgdl, "build_seconds": time.perf_counter() - start}
        write_json(manifest_path, manifest)
        return manifest

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        import numpy as np
        scores = np.zeros(self.weights.shape[0], dtype=np.float32)
        for token in set(self.analyzer(query)):
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            start, end = self.weights.indptr[term_id:term_id + 2]
            scores[self.weights.indices[start:end]] += self.weights.data[start:end]
        return [(i, float(scores[i])) for i in top_indices(scores, k, positive_only=True)]


def load_encoder(model_name: str, device: str, revision: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device, revision=revision, trust_remote_code=False)


def encoder_revision(encoder) -> str | None:
    return getattr(encoder[0].auto_model.config, "_commit_hash", None)


def build_dense(store: ChunkStore, directory: Path, model_name: str, device: str = "cpu",
                batch_size: int = 32, revision: str | None = None) -> dict:
    import numpy as np
    from tqdm import tqdm
    if batch_size <= 0 or len(store) == 0:
        raise ValueError("Embedding batch size and chunk count must be positive")
    params = {"chunk_fingerprint": store.manifest["fingerprint"], "model": model_name,
              "requested_revision": revision, "normalized": False, "dtype": "float32", "version": 1}
    identity = fingerprint(params)
    manifest_path = directory / "dense_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != identity:
            raise ValueError("Dense cache configuration changed")
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(model_name, device, revision)
    actual_revision = encoder_revision(encoder)
    dimension = encoder.get_sentence_embedding_dimension()
    progress_path = directory / "dense_progress.json"
    vectors_path = directory / "vectors.npy"
    n = len(store)
    completed = 0
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["fingerprint"] != identity or progress["actual_revision"] != actual_revision:
            raise ValueError("Partial embedding cache belongs to a different configuration/model revision")
        completed = progress["completed"]
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        if vectors.shape != (n, dimension):
            raise ValueError("Partial vector shape mismatch")
    else:
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype="float32", shape=(n, dimension))
    start = time.perf_counter()
    token_sample = []
    # SQLite row IDs are contiguous from zero and immutable in the chunk manifest.
    for offset in tqdm(range(completed, n, batch_size), desc="Encoding corpus"):
        rows = store.get(list(range(offset, min(offset + batch_size, n))))
        texts = [r["text"] for r in rows]
        encoded = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=False, show_progress_bar=False)
        if not np.isfinite(encoded).all():
            raise ValueError(f"Nonfinite embeddings at offset {offset}")
        vectors[offset:offset + len(rows)] = encoded
        # Sample from corpus only, not evaluation answers. Diagnose 256-token encoder truncation.
        if len(token_sample) < 1000:
            token_sample.extend(len(ids) for ids in encoder.tokenizer(texts, truncation=False)["input_ids"])
        vectors.flush()
        write_json(progress_path, {"fingerprint": identity, "actual_revision": actual_revision,
                                   "completed": offset + len(rows), "total": n})
    manifest = {**params, "fingerprint": identity, "actual_revision": actual_revision,
        "vectors": n, "dimension": dimension, "encoder_max_seq_length": encoder.max_seq_length,
        "resumed_from": completed, "this_session_encode_seconds": time.perf_counter() - start,
        "sampled_chunk_token_lengths": {"count": len(token_sample), "exceed_encoder_limit": sum(t > encoder.max_seq_length for t in token_sample),
                                        "mean": float(np.mean(token_sample)) if token_sample else None}}
    write_json(manifest_path, manifest)
    return manifest


class DenseIndex:
    def __init__(self, directory: Path, encoder, metric: str = "cosine"):
        import faiss
        import numpy as np
        if metric not in {"cosine", "l2"}:
            raise ValueError(metric)
        self.manifest = read_json(directory / "dense_manifest.json")
        self.encoder, self.metric = encoder, metric
        if encoder_revision(encoder) != self.manifest["actual_revision"]:
            raise ValueError("Query encoder revision differs from indexed encoder")
        vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        self.index = faiss.IndexFlatIP(vectors.shape[1]) if metric == "cosine" else faiss.IndexFlatL2(vectors.shape[1])
        for start in range(0, len(vectors), 8192):
            batch = np.array(vectors[start:start + 8192], dtype=np.float32, copy=True)
            if metric == "cosine":
                faiss.normalize_L2(batch)
            self.index.add(batch)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        import numpy as np
        vector = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=self.metric == "cosine", show_progress_bar=False)
        scores, ids = self.index.search(np.asarray(vector, dtype=np.float32), min(k, self.index.ntotal))
        return [(int(i), float(s) if self.metric == "cosine" else -float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


class Retriever:
    def __init__(self, store: ChunkStore, method: str, bm25: BM25Index | None = None,
                 dense: DenseIndex | None = None, candidate_k: int = 30, rrf_constant: int = 60):
        self.store, self.method = store, method
        self.bm25, self.dense = bm25, dense
        self.candidate_k, self.rrf_constant = candidate_k, rrf_constant
        if method not in {"bm25", "dense", "hybrid"}:
            raise ValueError(method)
        if method in {"bm25", "hybrid"} and bm25 is None or method in {"dense", "hybrid"} and dense is None:
            raise ValueError("Missing required index")
        for index in [bm25, dense]:
            if index and index.manifest["chunk_fingerprint"] != store.manifest["fingerprint"]:
                raise ValueError("Index belongs to a different chunk corpus")

    def search(self, question: str, k: int = 5) -> list[dict]:
        if self.method == "bm25":
            result = self.bm25.search(question, k)
        elif self.method == "dense":
            result = self.dense.search(question, k)
        else:
            result = reciprocal_rank_fusion([[i for i, _ in self.bm25.search(question, self.candidate_k)],
                [i for i, _ in self.dense.search(question, self.candidate_k)]], rank_constant=self.rrf_constant, k=k)
        rows = self.store.get([i for i, _ in result])
        return [{**row, "score": score, "rank": rank} for rank, (row, (_, score)) in enumerate(zip(rows, result), 1)]
