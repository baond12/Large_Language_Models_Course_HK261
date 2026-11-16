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
from evaluation_selection import load_selection, resolve_selection


def definitions(source):
    tree = ast.parse(source)
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def load_functions(source, namespace):
    exec(compile(ast.Module(body=definitions(source), type_ignores=[]), '<notebook functions>', 'exec'), namespace)
    return namespace


class NotebookPreparationTests(unittest.TestCase):
    def test_full_res_selection_matches_all_original_pairs_in_excel_order(self):
        ids = load_selection('full-res')
        self.assertEqual(len(ids), 2745)
        with tempfile.TemporaryDirectory(prefix='full_res_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            with patch.object(io, 'RUN', run), patch.object(io, 'RES_CSV', run/'evaluation.csv'), patch.object(io, 'EVALUATION_IDS', ids):
                records = io.prepare_evaluation()
            self.assertEqual([r['question_id'] for r in records], ids)
            self.assertEqual([r['excel_row'] for r in records], sorted(r['excel_row'] for r in records))
            questions = {r['id']:r['question'] for r in map(json.loads, (WORK/'data/questions.jsonl').read_text(encoding='utf-8').splitlines())}
            labels = {r['id']:r['answer'] for r in map(json.loads, (WORK/'data/labels.jsonl').read_text(encoding='utf-8').splitlines())}
            self.assertEqual(set(ids), set(questions))
            for row in records:
                self.assertEqual(row['Câu Hỏi'], questions[row['question_id']])
                self.assertEqual(row['Trả lời'], labels[row['question_id']])

    def test_full_res_prefix_is_not_complete_and_invalid_limit_is_rejected(self):
        ids, prefix, limited = resolve_selection('full-res', 100)
        self.assertTrue(limited)
        self.assertEqual(len(prefix), 100)
        self.assertEqual(len(ids), 2745)
        with self.assertRaisesRegex(ValueError, '--limit'):
            resolve_selection('full-res', 2746)
        self.assertEqual(resolve_selection('fixed-100')[1], io.CONTRACT['planned_evaluation_ids'])
        with tempfile.TemporaryDirectory(prefix='prefix_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            metrics = {name:1. for name in ['Cosine_Similarity','Jaccard_Similarity','Token_Overlap','BLEU_Score','ROUGE_L']}
            records = [{'id':uid,'answer':'a','reference':'a','error':None,'returned_text_starts_with_prompt':False,'backend_generation':{'effective_max_new_tokens':256},'generated_tokens':1,'metrics_notebook':metrics} for uid in prefix]
            (run/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
            with patch.object(io, 'RUN', run), patch.object(io, 'EVALUATION_IDS', prefix), patch.object(io, 'SELECTION_IDS', ids), patch.object(io, 'SELECTION_NAME', 'full-res'), patch.object(io, 'N', 100), patch.object(io, 'IS_PILOT', limited), redirect_stdout(StringIO()):
                io.finish([None]*100, metrics, lambda s:s)
            summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['eligible_for_final_evaluation'])
            self.assertEqual(summary['expected_selection_count'], 2745)

    def test_full_res_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='manifest_', dir=ROOT.parent/'qa') as tmp:
            code = Path(tmp)
            value = json.loads((ROOT/'full_res_selection.json').read_text(encoding='utf-8'))
            value['count'] -= 1
            (code/'full_res_selection.json').write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'count or ID digest'):
                load_selection('full-res', code)

    def test_full_evaluation_rejects_prompt_echo_and_unforwarded_limit(self):
        with tempfile.TemporaryDirectory(prefix='output_gate_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            metrics = {name: 1. for name in ['Cosine_Similarity', 'Jaccard_Similarity', 'Token_Overlap', 'BLEU_Score', 'ROUGE_L']}
            records = [{'id': uid, 'answer': 'a', 'reference': 'a', 'error': None, 'returned_text_starts_with_prompt': False, 'backend_generation': {'effective_max_new_tokens': 256}, 'generated_tokens': 1, 'metrics_notebook': metrics} for uid in io.CONTRACT['planned_evaluation_ids']]
            def finish():
                (run/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
                with patch.object(io, 'RUN', run), redirect_stdout(StringIO()):
                    io.finish([None]*100, metrics, lambda text: text)
                return json.loads((run/'summary.json').read_text(encoding='utf-8'))
            records[0]['returned_text_starts_with_prompt'] = True
            self.assertFalse(finish()['eligible_for_final_evaluation'])
            records[0]['returned_text_starts_with_prompt'] = False
            records[0]['backend_generation']['effective_max_new_tokens'] = None
            self.assertFalse(finish()['eligible_for_final_evaluation'])
            records[0]['backend_generation']['effective_max_new_tokens'] = 256
            self.assertTrue(finish()['eligible_for_final_evaluation'])

    def test_one_question_pilot_is_separate_from_full_evaluation(self):
        with tempfile.TemporaryDirectory(prefix='pilot_', dir=ROOT.parent/'qa') as tmp:
            run = Path(tmp)
            with patch.object(io, 'RUN', run), patch.object(io, 'RES_CSV', run/'evaluation.csv'), patch.object(io, 'N', 1), patch.object(io, 'IS_PILOT', True), patch.object(io, 'EVALUATION_IDS', io.CONTRACT['planned_evaluation_ids'][:1]):
                records = io.prepare_evaluation()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['question_id'], io.CONTRACT['planned_evaluation_ids'][0])
                scores = {'Cosine_Similarity': 0., 'Jaccard_Similarity': 0., 'Token_Overlap': 0., 'BLEU_Score': 0., 'ROUGE_L': 0.}
                item = {'id': records[0]['question_id'], 'answer': '', 'reference': 'a', 'metrics_notebook': scores, 'error': None, 'returned_text_starts_with_prompt': False}
                (run/'predictions.jsonl').write_text(json.dumps(item)+'\n', encoding='utf-8')
                with redirect_stdout(StringIO()):
                    io.finish([None], scores, lambda s:s)
                summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
                self.assertTrue(summary['limited_run'])
                self.assertFalse(summary['eligible_for_final_evaluation'])

    def test_reusing_modified_index_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='reuse_', dir=ROOT.parent/'qa') as tmp:
            root = Path(tmp); old = root/'old'; new = root/'new'
            old.mkdir(); new.mkdir(); (old/'faiss_db').mkdir()
            env = {'source_sha256': {'input':'same'}, 'model_revisions': {'embedding':'same'}, 'distance_branch':'notebook', 'index_logic_sha256':'same', 'packages':{}}
            for folder in [old,new]:
                io.write_json(folder/'environment.json', env)
            for name in ['index.faiss','index.pkl']:
                (old/'faiss_db'/name).write_bytes(b'original index fixture')
            io.write_json(old/'index_info.json', {'index_files_sha256':{name:io.sha(old/'faiss_db'/name) for name in ['index.faiss','index.pkl']}})
            io.write_json(old/'document_order.json', ['a.txt'])
            with patch.object(io, 'RUN', new), patch.object(io, 'INDEX_DIR', new/'faiss_db'), patch.dict(io.os.environ, {'BASELINE_REUSE_INDEX_FROM':str(old)}), redirect_stdout(StringIO()):
                io.reuse_index()
                self.assertEqual(io.INDEX_DIR, old/'faiss_db')
                (old/'faiss_db/index.pkl').write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError, 'modified index file'):
                    io.reuse_index()

    def test_original_prompt_retrieval_and_declared_generation_fix(self):
        original = json.loads(ORIGINAL.read_text(encoding='utf-8'))
        for mode in ['notebook', 'cosine']:
            build(mode)
            derived = json.loads((ROOT / f'Baseline_Docker_{mode}.ipynb').read_text(encoding='utf-8'))
            self.assertTrue(all(not c.get('outputs') for c in derived['cells']))
            assignments = [node for node in ast.parse(''.join(derived['cells'][5]['source'])).body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'llm' for t in node.targets)]
            class Loader:
                @staticmethod
                def from_model_id(**kwargs):
                    return kwargs
            namespace = {'HuggingFacePipeline': Loader, 'baseline_io': SimpleNamespace(LLM_REVISION='frozen-revision')}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), '<generated LLM call>', 'exec'), namespace)
            self.assertEqual(namespace['llm']['model_id'], 'Qwen/Qwen1.5-1.8B')
            self.assertEqual(namespace['llm']['model_kwargs'], {'temperature': 0, 'max_length': 1024, 'revision': 'frozen-revision'})
            self.assertEqual(namespace['llm']['pipeline_kwargs']['framework'], 'pt')
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
            self.assertEqual(calls[0], calls_before[0])
            self.assertEqual(calls[-1][1], calls_before[-1][1])
            self.assertEqual(answer_after, answer_before)
            self.assertEqual(calls[-1][2], {'pipeline_kwargs': {'max_new_tokens': 256, 'return_full_text': False}})
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
