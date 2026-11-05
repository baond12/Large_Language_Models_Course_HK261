from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEMP = ROOT / ".cache" / "test-temp"
TEMP.mkdir(parents=True, exist_ok=True)
assert TEMP.resolve().is_relative_to(ROOT.resolve())

from rag_assignment.common import check_under_4b, fingerprint, read_json, write_json, write_jsonl
from rag_assignment.data import canonical_law_id, filename_law_id, law_ids_in_text, make_splits
from rag_assignment.generation import build_prompt, check_citations
from rag_assignment.metrics import lexical_metrics, paired_bootstrap, retrieval_metrics
from rag_assignment.retrieval import BM25Index, Retriever, reciprocal_rank_fusion


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


def passages():
    return [{"chunk_id": str(i), "source": f"doc{i}.txt", "text": "Nội dung. " * 150, "section": f"Điều {i}"} for i in range(1, 6)]


class MetricTests(unittest.TestCase):
    def test_identical(self):
        self.assertTrue(all(x == 1.0 for x in lexical_metrics("Luật lao động 2024.", "Luật lao động 2024.").values()))

    def test_empty(self):
        for a, b in [("", "a"), ("a", ""), ("", "")]:
            self.assertTrue(all(x == 0.0 for x in lexical_metrics(a, b).values()))

    def test_hand_computed_overlap(self):
        x = lexical_metrics("a a b", "a b b")
        self.assertEqual(x["Jaccard_Similarity"], 1)
        self.assertAlmostEqual(x["Token_Overlap"], 2 / 3)
        self.assertAlmostEqual(x["BLEU_Score"], 2 / 3)
        self.assertEqual(x["Token_Overlap_Set_PDF"], 1)
        self.assertAlmostEqual(x["ROUGE_L"], 2 / 3)

    def test_brevity_penalty(self):
        self.assertAlmostEqual(lexical_metrics("a", "a b")["BLEU_Score"], math.exp(-1))

    def test_order_matters_for_rouge(self):
        x = lexical_metrics("a b c", "c b a")
        self.assertEqual(x["Jaccard_Similarity"], 1)
        self.assertAlmostEqual(x["ROUGE_L"], 1 / 3)

    def test_vietnamese_normalization_matches_baseline(self):
        self.assertEqual(lexical_metrics("ĐIỀU 1: tiền, lương!", "điều 1 tiền lương")["ROUGE_L"], 1)

    @unittest.skipUnless((ROOT.parent / "Baseline.ipynb").exists(), "Original baseline not present in standalone source package")
    def test_metrics_match_original_notebook(self):
        import numpy as np
        original = json.loads((ROOT.parent / "Baseline.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in original["cells"] if c["cell_type"] == "code")
        names = {"normalize_text", "jaccard_similarity", "token_overlap_score", "bleu_score_simple", "rouge_l_score"}
        nodes = [n for n in ast.parse(code.replace("!", "#!")).body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = {"re": __import__("re"), "np": np, "Counter": Counter}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_baseline_functions", "exec"), env)
        mapping = {"Jaccard_Similarity": "jaccard_similarity", "Token_Overlap": "token_overlap_score",
                   "BLEU_Score": "bleu_score_simple", "ROUGE_L": "rouge_l_score"}
        rng = random.Random(261)
        for _ in range(100):
            a = " ".join(rng.choices(["điều", "luật", "1", "điều", "quyền"], k=rng.randrange(15)))
            b = " ".join(rng.choices(["điều", "luật", "1", "điều", "quyền"], k=rng.randrange(15)))
            actual = lexical_metrics(a, b)
            for metric, name in mapping.items():
                self.assertAlmostEqual(actual[metric], env[name](a, b), places=12)

    def test_bootstrap_deterministic(self):
        self.assertEqual(paired_bootstrap([.1, .2, -.1], samples=100), paired_bootstrap([.1, .2, -.1], samples=100))

    def test_retrieval_duplicates_not_double_counted(self):
        x = retrieval_metrics(["a", "a", "x", "b"], ["a", "b"], 5)
        self.assertEqual(x["Recall@5"], 1)
        self.assertEqual(x["MRR@5"], 1)


class DataTests(unittest.TestCase):
    def test_canonical_legal_id(self):
        self.assertEqual(canonical_law_id("50/2024/TT-NHNN"), filename_law_id("export_1/50_2024_TT-NHNN_2.txt"))

    def test_national_assembly_reference(self):
        self.assertEqual(law_ids_in_text("Luật số 41/2024/QH15"), ["412024QH15"])

    def test_multiple_reference_ids(self):
        self.assertEqual(law_ids_in_text("50/2024/TT-NHNN; 41/2024/QH15"), ["412024QH15", "502024TTNHNN"])

    def test_unknown_label_not_a_document(self):
        self.assertEqual(law_ids_in_text("Không trích xuất được"), [])

    def test_spaces_around_reference_slash(self):
        self.assertEqual(law_ids_in_text("Số: 01 / 2015 / QĐ-UBND"), ["012015QDUBND"])

    def test_split_group_isolation_and_determinism(self):
        records = [{"id": f"q{i}", "law_ids": [f"law{i // 3}"]} for i in range(60)]
        a = make_splits(records, test_size=10, dev_size=10)
        b = make_splits(records, test_size=10, dev_size=10)
        self.assertEqual(a, b)
        owners = {}
        by_id = {r["id"]: r for r in records}
        for split, ids in a["ids"].items():
            for uid in ids:
                for law in by_id[uid]["law_ids"]:
                    self.assertIn(owners.setdefault(law, split), [split])

    def test_connected_multilabel_groups(self):
        records = [{"id": "q0", "law_ids": ["a"]}, {"id": "q1", "law_ids": ["a", "b"]},
                   {"id": "q2", "law_ids": ["b"]}] + [{"id": f"q{i}", "law_ids": []} for i in range(3, 20)]
        split = make_splits(records, test_size=3, dev_size=3)
        locations = [s for s, ids in split["ids"].items() if "q0" in ids or "q1" in ids or "q2" in ids]
        self.assertEqual(len(locations), 1)

    def test_diagnostics_do_not_overlap_main(self):
        records = [{"id": f"q{i}", "law_ids": [f"law{i}"]} for i in range(60)]
        split = make_splits(records, test_size=10, dev_size=10, corpus_ids={f"law{i}" for i in range(60)})
        all_ids = [q for ids in split["ids"].values() for q in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertTrue(split["ids"]["diagnostic_dev"])
        self.assertTrue(split["ids"]["diagnostic_test"])

    def test_model_parameter_boundary(self):
        check_under_4b(3_999_999_999)
        for count in [0, 4_000_000_000, 4_100_000_000]:
            with self.assertRaises(ValueError):
                check_under_4b(count)

    @unittest.skipUnless((ROOT / "data/manifest.json").exists(), "Full dataset not prepared")
    def test_prepared_data_split_and_inference_schema(self):
        questions = [json.loads(l) for l in (ROOT / "data/questions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(set(q) == {"id", "question"} for q in questions))
        split = read_json(ROOT / "data/splits.json")
        self.assertGreaterEqual(len(split["ids"]["test"]), 100)
        all_ids = [q for ids in split["ids"].values() for q in ids]
        self.assertEqual(len(all_ids), len(questions))
        self.assertEqual(len(all_ids), len(set(all_ids)))


class ContextTests(unittest.TestCase):
    def test_prompt_keeps_question_and_fits(self):
        question = "Câu hỏi rất quan trọng?"
        prompt, metadata = build_prompt(question, passages(), CharTokenizer(), 1200, "grounded", "balanced")
        self.assertIn(question, prompt)
        self.assertLessEqual(metadata["input_tokens"], 1200)
        self.assertTrue(all(p["included_tokens"] > 0 for p in metadata["passages"]))

    def test_prefix_vs_balanced(self):
        _, a = build_prompt("Hỏi?", passages(), CharTokenizer(), 1000, "baseline", "prefix")
        _, b = build_prompt("Hỏi?", passages(), CharTokenizer(), 1000, "baseline", "balanced")
        self.assertEqual(sum(p["included_tokens"] > 0 for p in a["passages"]), 1)
        self.assertEqual(sum(p["included_tokens"] > 0 for p in b["passages"]), 5)

    def test_long_question_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            build_prompt("Q" * 3000, passages(), CharTokenizer(), 1000)

    def test_no_context(self):
        prompt, metadata = build_prompt("Hỏi?", [], CharTokenizer(), 1000, "grounded", "balanced")
        self.assertIn("Hỏi?", prompt)
        self.assertEqual(metadata["passages"], [])

    def test_unknown_source_label_flagged(self):
        _, meta = build_prompt("Hỏi?", passages(), CharTokenizer(), 1000, "baseline", "prefix")
        self.assertEqual(check_citations("Theo [S1] và [S99]", meta)["invalid_cited_labels"], [99])


class SparseTests(unittest.TestCase):
    def test_rrf_hand_computation_and_ties(self):
        result = reciprocal_rank_fusion([[3, 7], [7, 3]], k=2)
        self.assertEqual([i for i, _ in result], [3, 7])
        self.assertAlmostEqual(result[0][1], 1 / 61 + 1 / 62)

    def test_rrf_duplicates_not_inflated(self):
        result = reciprocal_rank_fusion([[3, 3]], k=2)
        self.assertEqual(result, [(3, 1 / 61)])

    def test_bm25_real_index_roundtrip(self):
        from rag_assignment.chunks import ChunkStore
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory = Path(temp)
            db = sqlite3.connect(directory / "chunks.sqlite")
            db.execute("CREATE TABLE chunks (row_id INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT, source TEXT, law_id TEXT, section TEXT, text TEXT)")
            for i, text in enumerate(["tiền lương người lao động", "ngân hàng lãi suất", "tiền lương tiền lương"]):
                db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", (i, str(i), str(i), str(i), str(i), "", text))
            db.commit()
            db.close()
            write_json(directory / "chunks_manifest.json", {"fingerprint": "toy", "chunks": 3})
            store = ChunkStore(directory)
            BM25Index.build(store, directory / "bm25")
            index = BM25Index(directory / "bm25")
            self.assertEqual(index.search("ngân hàng")[0][0], 1)
            self.assertEqual(index.search("zzzzunknown"), [])
            result = Retriever(store, "bm25", bm25=index).search("người lao động", 2)
            self.assertEqual(result[0]["source"], "0")
            store.close()


if __name__ == "__main__":
    unittest.main()
