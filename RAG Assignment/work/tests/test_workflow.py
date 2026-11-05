"""Workflow invariants with synthetic data. These results are tests, not benchmark scores."""
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
TEMP = ROOT / ".cache/test-temp"
TEMP.mkdir(parents=True, exist_ok=True)
assert TEMP.resolve().is_relative_to(ROOT.resolve())

from rag_assignment.common import read_json, read_jsonl, write_json, write_jsonl
from rag_assignment.data import prepare
from repair_jsonl import repair


class PrepareIntegrationTests(unittest.TestCase):
    def test_actual_xlsx_zip_to_atomic_data_and_chunks(self):
        import openpyxl
        from rag_assignment.chunks import build_chunks, ChunkStore
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            root = Path(temp)
            inputs = root / "inputs"
            inputs.mkdir()
            book = openpyxl.Workbook()
            sheet = book.active
            sheet.append(["Câu Hỏi", "Trả lời", "Số hiệu VBPL (Trích xuất)", "Câu trả lời của URAx-Law"])
            for i in range(30):
                sheet.append([f"Câu hỏi {i}?", f"Đáp án {i} không được index", f"{i}/2024/TT-BTC", "secret-other-answer"])
            book.save(inputs / "RES.xlsx")
            with zipfile.ZipFile(inputs / "Dataset.zip", "w") as archive:
                for i in range(30):
                    archive.writestr(f"export_1/{i}_2024_TT-BTC.txt", f"Số: {i}/2024/TT-BTC\r\nĐiều 1. Đối tượng\r\nNội dung corpus độc lập {i}.")
            (inputs / "Baseline.ipynb").write_text("{}", encoding="utf-8")
            (inputs / "RAG_Assignment.pdf").write_bytes(b"synthetic hash-only test input")
            out = root / "data"
            audit = prepare(inputs, out, test_size=5, dev_size=5)
            self.assertEqual(audit["documents"], 30)
            self.assertEqual(audit["questions"], 30)
            self.assertEqual(prepare(inputs, out, test_size=5, dev_size=5), audit)
            self.assertFalse(list(out.glob("corpus.building-*.sqlite")))
            build_chunks(out, root / "index")
            store = ChunkStore(root / "index")
            self.assertEqual(len(store), 30)
            self.assertFalse(any("Đáp án" in t or "secret-other-answer" in t for t in store.texts()))
            store.close()
            with self.assertRaises(ValueError):
                prepare(inputs, out, seed=999, test_size=5, dev_size=5)


class RecoveryTests(unittest.TestCase):
    def test_repair_only_final_incomplete_record(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            path = Path(temp) / "predictions.jsonl"
            original = b'{"id":"ok","error":"real failure"}\n{"id":'
            path.write_bytes(original)
            result = repair(path)
            self.assertEqual(result["status"], "removed_incomplete_final_line")
            self.assertEqual(path.with_name(path.name + ".incomplete.bak").read_bytes(), original)
            self.assertEqual(read_jsonl(path), [{"id": "ok", "error": "real failure"}])

    def test_repair_rejects_middle_corruption(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            path = Path(temp) / "predictions.jsonl"
            original = b'broken\n{"id":"ok"}\n'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                repair(path)
            self.assertEqual(path.read_bytes(), original)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch needed by inference runner")
class ExperimentIntegrationTests(unittest.TestCase):
    def test_resume_scoring_comparison_and_error_denominator(self):
        import numpy as np
        from rag_assignment.experiment import run_generation, score_run, compare_runs, select_candidate

        class FakeStore:
            def close(self):
                pass

        class FakeRetriever:
            store = FakeStore()

            def search(self, question, k):
                return [{"chunk_id": "c", "source": "corpus.txt", "text": "chỉ có corpus"}]

        class FakeGenerator:
            info = {"parameters": 123, "model": "toy"}

            def __init__(self, *args, **kwargs):
                pass

            def generate(self, question, passages, *args):
                if question == "bad":
                    raise RuntimeError("synthetic generation failure")
                return {"answer": "a b", "generation_seconds": .01}

        class FakeEvaluator:
            max_seq_length = 256

            def encode(self, texts, **kwargs):
                return np.array([[len(t), 1.] for t in texts])

        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            root = Path(temp)
            data = root / "data"
            write_json(data / "manifest.json", {"fingerprint": "toy"})
            write_json(data / "splits.json", {"ids": {"dev": ["q1", "q2"]}})
            write_jsonl(data / "questions.jsonl", [{"id": "q1", "question": "good"}, {"id": "q2", "question": "bad"}])
            write_jsonl(data / "labels.jsonl", [{"id": "q1", "answer": "a b", "law_ids": []}, {"id": "q2", "answer": "a", "law_ids": []}])
            write_json(root / "index/chunks_manifest.json", {"fingerprint": "toy"})
            write_json(root / "index/bm25/bm25_manifest.json", {"fingerprint": "bm25-toy"})
            write_json(root / "models.lock.json", {"toy": {"revision": "abc"}, "keepitreal/vietnamese-sbert": {"revision": "def"}})
            config = {"name": "toy", "index_dir": "index", "retriever": "bm25", "llm": "toy"}
            folder = root / "run"
            with patch("rag_assignment.experiment.make_retriever", return_value=FakeRetriever()), patch("rag_assignment.generation.LocalGenerator", FakeGenerator):
                result = run_generation(config, root, data, "dev", folder, "cpu")
                self.assertEqual(result["questions"], 2)
                again = run_generation(config, root, data, "dev", folder, "cpu")
                self.assertEqual(again["status"], "already_complete")
                self.assertEqual(len(read_jsonl(folder / "predictions.jsonl")), 2)
                with self.assertRaises(ValueError):
                    run_generation({**config, "k": 3}, root, data, "dev", folder, "cpu")
            with patch("rag_assignment.experiment.load_encoder", return_value=FakeEvaluator()):
                summary = score_run(folder, data, root)
            self.assertEqual(summary["errors"], 1)
            self.assertEqual(summary["questions"], 2)
            self.assertAlmostEqual(summary["metrics"]["ROUGE_L"], .5)
            compared = compare_runs(folder, folder, root / "comparison", minimum=2, required_split="dev")
            self.assertFalse(compared["shows_any_primary_metric_gain"])
            with self.assertRaises(ValueError):
                compare_runs(folder, folder, root / "wrong_split", minimum=2)
            with self.assertRaises(ValueError):
                compare_runs(folder, folder, root / "too_small", minimum=100, required_split="dev")
            choice = select_candidate([folder], root / "selection.json")
            self.assertEqual(choice["selected_config"]["name"], "toy")
            # A modified score file must not silently influence comparisons.
            with (folder / "scores.jsonl").open("a", encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                compare_runs(folder, folder, root / "tampered", minimum=2, required_split="dev")


if __name__ == "__main__":
    unittest.main()
