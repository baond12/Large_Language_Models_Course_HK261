"""Real library integration tests with tiny local random weights; no model downloads."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEMP = ROOT / ".cache" / "test-temp"
TEMP.mkdir(parents=True, exist_ok=True)
assert TEMP.resolve().is_relative_to(ROOT.resolve())


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ["torch", "transformers", "tokenizers"]), "Model dependencies not installed")
class GeneratorIntegrationTests(unittest.TestCase):
    def test_actual_chat_generation_preserves_eos_list_and_budget(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from rag_assignment.generation import LocalGenerator
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory = Path(temp)
            backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "luật": 3, "nguồn": 4}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
            tokenizer.save_pretrained(directory)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=256, n_embd=16, n_layer=1, n_head=2,
                                             eos_token_id=[1, 2], pad_token_id=2, bos_token_id=1))
            model.save_pretrained(directory)
            generator = LocalGenerator(str(directory), device="cpu", max_input_tokens=200, max_new_tokens=4, chat_template=True)
            self.assertEqual(generator.tokenizer.pad_token_id, 2)
            self.assertEqual(generator.info["input_format"], "tokenizer_chat_template")
            with patch.object(generator.model, "generate", wraps=generator.model.generate) as generate:
                result = generator.generate("luật", [{"chunk_id": "c", "source": "s", "text": "nguồn", "section": ""}])
                self.assertEqual(generate.call_args.kwargs["eos_token_id"], [1, 2])
            self.assertTrue(result["context"]["chat_template"])
            self.assertLessEqual(result["context"]["input_tokens"], 200)
            self.assertLessEqual(result["generated_tokens"], 4)
            self.assertNotIn("Câu hỏi:", result["answer"])

    def test_actual_generate_slices_completion_only(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from rag_assignment.generation import LocalGenerator
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory = Path(temp)
            tokenizer_backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "luật": 3, "nguồn": 4}, unk_token="[UNK]"))
            tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
            tokenizer.save_pretrained(directory)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=256, n_embd=16, n_layer=1, n_head=2,
                                             eos_token_id=1, pad_token_id=2, bos_token_id=1))
            model.save_pretrained(directory)
            generator = LocalGenerator(str(directory), device="cpu", max_input_tokens=200, max_new_tokens=4)
            result = generator.generate("luật", [{"chunk_id": "c", "source": "s", "text": "nguồn", "section": ""}])
            self.assertLessEqual(result["generated_tokens"], 4)
            self.assertGreater(result["context"]["input_tokens"], result["generated_tokens"])
            self.assertNotIn("Câu hỏi:", result["answer"])
            self.assertLess(generator.info["parameters"], 4_000_000_000)
            self.assertEqual(generator.info["input_format"], "plain_completion")
            self.assertEqual(generator.stop_token_ids, 1)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS not installed")
class DenseIntegrationTests(unittest.TestCase):
    def test_l2_and_cosine_differ_as_expected(self):
        import numpy as np
        from rag_assignment.common import write_json
        from rag_assignment.retrieval import DenseIndex

        class Config:
            _commit_hash = "toy_revision"

        class Encoder:
            def __getitem__(self, index):
                class First:
                    class auto_model:
                        config = Config()
                return First()

            def encode(self, texts, **kwargs):
                # Unit vector unchanged by requested normalization.
                return np.array([[1., 0.]], dtype=np.float32)

        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory = Path(temp)
            np.save(directory / "vectors.npy", np.array([[10., 0.], [1., 1.]], dtype=np.float32))
            write_json(directory / "dense_manifest.json", {"actual_revision": "toy_revision"})
            l2, cosine = DenseIndex(directory, Encoder(), "l2"), DenseIndex(directory, Encoder(), "cosine")
            self.assertEqual(l2.search("query")[0][0], 1)
            self.assertEqual(cosine.search("query")[0][0], 0)
