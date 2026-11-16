"""CPU checks of notebook preservation and real evaluation selection, not GPU RAG tests."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import notebook_io as io
from prepare_notebooks import ORIGINAL, ROOT, WORK, build


def definitions(source):
    tree = ast.parse(source)
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def load_functions(source, namespace):
    exec(compile(ast.Module(body=definitions(source), type_ignores=[]), '<notebook functions>', 'exec'), namespace)
    return namespace


class NotebookPreparationTests(unittest.TestCase):
    def test_original_prompt_generate_call_and_raw_return_are_preserved(self):
        original = json.loads(ORIGINAL.read_text(encoding='utf-8'))
        for mode in ['notebook', 'cosine']:
            build(mode)
            derived = json.loads((ROOT / f'Baseline_Docker_{mode}.ipynb').read_text(encoding='utf-8'))
            self.assertTrue(all(not c.get('outputs') for c in derived['cells']))
            calls = []
            observed = []
            class Retriever:
                def similarity_search(self, question, k):
                    calls.append(('retrieve', question, k))
                    return [SimpleNamespace(page_content='first source'), SimpleNamespace(page_content='second source')]
            class LLM:
                def generate(self, prompts, **kwargs):
                    calls.append(('generate', prompts, kwargs))
                    # A wrapper returning the prompt must not be silently sliced by the adapter.
                    return SimpleNamespace(generations=[[SimpleNamespace(text=prompts[0]+'RAW COMPLETION')]])
            common = {'vectorstore': Retriever(), 'llm': LLM(), 'baseline_io': SimpleNamespace(capture_context=lambda *args: observed.append(args))}
            source_original = ''.join(original['cells'][5]['source'])
            source_derived = ''.join(derived['cells'][5]['source'])
            before = load_functions(source_original, dict(common))
            after = load_functions(source_derived, dict(common))
            with redirect_stdout(StringIO()):
                answer_before = before['rag_query']('Test question')
                calls_before = deepcopy(calls); calls.clear()
                answer_after = after['rag_query']('Test question')
            self.assertEqual(calls, calls_before)
            self.assertEqual(answer_after, answer_before)
            self.assertEqual(calls[-1][2], {'max_new_tokens': 256})
            self.assertEqual(observed[0][2], calls[-1][1][0])

    def test_real_RES_selection_has_exact_same_100_ids_and_questions(self):
        with tempfile.TemporaryDirectory(prefix='selection_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            with patch.object(io, 'RUN', run), patch.object(io, 'RES_CSV', run/'evaluation.csv'):
                records = io.prepare_evaluation()
            self.assertEqual(len(records), 100)
            self.assertEqual([r['question_id'] for r in records], io.CONTRACT['planned_evaluation_ids'])
            questions = {r['id']: r['question'] for r in map(json.loads, (WORK/'data/questions.jsonl').read_text(encoding='utf-8').splitlines())}
            labels = {r['id']: r for r in map(json.loads, (WORK/'data/labels.jsonl').read_text(encoding='utf-8').splitlines())}
            for row in records:
                self.assertEqual(row['Câu Hỏi'], questions[row['question_id']])
                self.assertEqual(row['Trả lời'], labels[row['question_id']]['answer'])
            result = {'scope': 'local RES evaluation selection only', 'pairs': len(records), 'same_ids_questions_references_as_fixed_selection': True, 'gpu_experiment_run': False}
            (ROOT.parent/'qa/evaluation_selection_validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')

    def test_zip_path_escape_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='archive_', dir=ROOT.parent/'qa') as tmp:
            root = Path(tmp); data = root/'inputs'; data.mkdir()
            with zipfile.ZipFile(data/'Dataset.zip', 'w') as z:
                z.writestr('../export_1/escape.txt', 'must not escape')
            with patch.object(io, 'DATA_ROOT', data), patch.object(io, 'CORPUS_ROOT', root/'corpus'), patch.object(io, 'prepare_evaluation', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                    io.prepare_inputs()
            self.assertFalse((root/'export_1/escape.txt').exists())

    def test_metrics_keep_zero_answers_in_denominator_and_separate_PDF(self):
        from collections import Counter
        import re
        original = json.loads(ORIGINAL.read_text(encoding='utf-8'))
        functions = load_functions(''.join(original['cells'][5]['source']), {'Counter': Counter, 're': re})
        pred, ref = 'a a b', 'a b b'
        counter_overlap = functions['token_overlap_score'](pred, ref)
        self.assertAlmostEqual(counter_overlap, 2/3)
        with tempfile.TemporaryDirectory(prefix='metrics_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            records = []
            for i, uid in enumerate(io.CONTRACT['planned_evaluation_ids']):
                metrics = {'Cosine_Similarity': 1.0 if i == 0 else 0.0, 'Jaccard_Similarity': 1.0 if i == 0 else 0.0, 'Token_Overlap': counter_overlap if i == 0 else 0.0, 'BLEU_Score': 2/3 if i == 0 else 0.0, 'ROUGE_L': 2/3 if i == 0 else 0.0}
                records.append({'id': uid, 'answer': pred if i == 0 else '', 'reference': ref, 'error': None if i == 0 else 'deliberate fixture error', 'returned_text_starts_with_prompt': False, 'metrics_notebook': metrics})
            (run/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
            means = {key: sum(r['metrics_notebook'][key] for r in records)/100 for key in records[0]['metrics_notebook']}
            with patch.object(io, 'RUN', run), redirect_stdout(StringIO()):
                io.finish([None]*100, means, functions['normalize_text'])
            summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['questions'], 100)
            self.assertEqual(summary['errors'], 99)
            self.assertAlmostEqual(summary['metrics_pdf']['Token_Overlap_Set_PDF'], .01)
            self.assertAlmostEqual(summary['metrics_pdf']['BLEU_Set_PDF'], .01)
            self.assertAlmostEqual(summary['metrics_notebook']['Token_Overlap'], (2/3)/100)
            self.assertFalse(summary['rerun_independently_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
