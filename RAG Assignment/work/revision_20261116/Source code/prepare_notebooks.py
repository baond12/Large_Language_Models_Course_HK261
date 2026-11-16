"""Derive runnable notebooks by explicit edits to the instructor-provided cells."""
from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = ROOT.parent
WORK = REVISION.parent
ORIGINAL = WORK.parent / 'Baseline.ipynb'
EXPECTED_SHA = '2dc1acfb86465aa4c53e3205187c5c6999192ae6cf02643a98ee3a103b935572'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one original fragment, got {source.count(old)}: {old[:100]}')
    return source.replace(old, new, 1)


def build(mode):
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    original = json.loads(raw)
    nb = deepcopy(original)
    sources = [''.join(c.get('source', [])) for c in original['cells']]
    updated = list(sources)
    updated[0] = '''# Operational setup only. No pip upgrades or model logic changes here.
import os
import sys
from pathlib import Path
_roots = [Path.cwd(), *Path.cwd().parents]
_work = next((p for p in _roots if (p / "revision_20261116/Source code/notebook_io.py").is_file()), None)
if _work is None:
    raise RuntimeError("Open from the course work directory or a child directory")
sys.path.insert(0, str(_work / "revision_20261116/Source code"))
import notebook_io as baseline_io
baseline_io.begin()
'''
    updated[0] = updated[0].replace('import notebook_io as baseline_io', f'os.environ["BASELINE_DISTANCE"] = "{mode}"\nimport notebook_io as baseline_io')
    updated[1] = replace_once(sources[1], '!pip install faiss-cpu\n', '')
    updated[2] = '''# Replace Colab/Drive and live Google Sheets loading with the supplied local inputs.
import pandas as pd
import os
import zipfile
DOCUMENTS_PATH = baseline_io.prepare_inputs()
df = pd.read_csv(baseline_io.RES_CSV)
print(f"Fixed evaluation pairs: {len(df)}")
'''
    updated[3] = replace_once(sources[3], 'DOCUMENTS_PATH = "/content/export_1"', 'DOCUMENTS_PATH = str(baseline_io.CORPUS_ROOT)')
    updated[3] = replace_once(updated[3], 'OUTPUT_DIR = "faiss_db"', 'OUTPUT_DIR = str(baseline_io.INDEX_DIR)')
    updated[3] = replace_once(updated[3], 'for filename in os.listdir(path):', 'for filename in baseline_io.document_filenames(path):')
    updated[3] = replace_once(updated[3], 'model_kwargs={"device": "cuda"}', 'model_kwargs={"device": "cuda", "revision": baseline_io.EMBEDDING_REVISION}')
    updated[3] += '\nbaseline_io.record_index(vectorstore, len(raw_documents), len(texts))\n'
    updated[4] = replace_once(sources[4], 'VECTORSTORE_PATH = "faiss_db"', 'VECTORSTORE_PATH = str(baseline_io.INDEX_DIR)')
    updated[4] = replace_once(updated[4], 'model_kwargs={"device": "cpu"}', 'model_kwargs={"device": "cpu", "revision": baseline_io.EMBEDDING_REVISION}')
    updated[5] = replace_once(sources[5], 'from langchain_google_genai import ChatGoogleGenerativeAI\n', '# Unused Google API import omitted; this notebook does not call an external LLM API.\n')
    updated[5] = replace_once(updated[5], 'df = pd.read_csv("res.csv")', 'df = pd.read_csv(baseline_io.RES_CSV)')
    updated[5] = replace_once(updated[5], 'df = df.head(30)', 'df = df.head(baseline_io.N)\nassert len(df) == baseline_io.N and (len(df) >= 100 or baseline_io.IS_PILOT)')
    updated[5] = replace_once(updated[5], '    "faiss_db",', '    str(baseline_io.INDEX_DIR),')
    updated[5] = replace_once(updated[5], 'model_kwargs={"temperature": 0, "max_length": 1024}', 'model_kwargs={"temperature": 0, "max_length": 1024, "revision": baseline_io.LLM_REVISION}')
    updated[5] = replace_once(updated[5], '# ==================== HELPER FUNCTIONS ====================', 'baseline_io.record_model(llm, embeddings)\n\n# ==================== HELPER FUNCTIONS ====================')
    updated[5] = replace_once(updated[5], '    output = llm.generate([prompt],  max_new_tokens=256)', '    baseline_io.capture_context(question, docs, prompt)\n    output = llm.generate([prompt],  max_new_tokens=256)')
    updated[5] = replace_once(updated[5], '    except Exception as e:\n', '    except Exception as e:\n        baseline_io.capture_error(question)\n')
    updated[5] = replace_once(updated[5], '    results.append(metrics)', '    baseline_io.record_result(idx, row, response, metrics)\n    results.append(metrics)')
    updated[5] = replace_once(updated[5], 'results_df.to_csv("rag_baseline_metrics.csv", index=False)', 'results_df.to_csv(baseline_io.RUN / "rag_baseline_metrics.csv", index=False)')
    updated[5] += '\nbaseline_io.finish(results_df, mean_metrics, normalize_text)\n'
    if mode == 'cosine':
        updated[3] = 'from langchain_community.vectorstores.utils import DistanceStrategy\n' + updated[3]
        updated[3] = replace_once(updated[3], '            embeddings\n        )', '            embeddings,\n            normalize_L2=True,\n            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT\n        )')
        for i in [4, 5]:
            updated[i] = replace_once(updated[i], '    allow_dangerous_deserialization=True\n)', '    allow_dangerous_deserialization=True,\n    normalize_L2=True,\n    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT\n)')
    # Skip only index construction when an index from this same notebook run is verified.
    prefix = 'from langchain_community.vectorstores.utils import DistanceStrategy\n' if mode == 'cosine' else ''
    updated[3] = prefix + 'if os.environ.get("BASELINE_REUSE_INDEX_FROM"):\n    baseline_io.reuse_index()\nelse:\n' + ''.join('    ' + line if line.strip() else line for line in updated[3].splitlines(keepends=True))
    for i, cell in enumerate(nb['cells']):
        cell['source'] = updated[i].splitlines(keepends=True)
        cell['execution_count'] = None
        cell['outputs'] = []
        ast.parse(updated[i])
    nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'baseline_provenance': {'original_sha256': EXPECTED_SHA, 'distance_branch': mode, 'execution_status': 'not_executed', 'source': 'instructor notebook with explicit cell edits'}}
    # Strip only the added observation call, then require exact function AST equality.
    original_functions = {x.name: x for x in ast.parse(sources[5]).body if isinstance(x, ast.FunctionDef)}
    derived_functions = {x.name: x for x in ast.parse(updated[5]).body if isinstance(x, ast.FunctionDef)}
    class RemoveObserver(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'baseline_io':
                return None
            return self.generic_visit(node)
    for name, fn in original_functions.items():
        actual = RemoveObserver().visit(deepcopy(derived_functions[name]))
        assert ast.dump(fn, include_attributes=False) == ast.dump(actual, include_attributes=False), name
    path = ROOT / f'Baseline_Docker_{mode}.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    diff = []
    for i, (before, after) in enumerate(zip(sources, updated)):
        before_lines = (before.rstrip('\n') + '\n').splitlines(keepends=True)
        after_lines = (after.rstrip('\n') + '\n').splitlines(keepends=True)
        diff.extend(difflib.unified_diff(before_lines, after_lines, fromfile=f'original/cell_{i}', tofile=f'{mode}/cell_{i}'))
    (ROOT / f'changes_{mode}.diff').write_text(''.join(diff), encoding='utf-8')
    return {'mode': mode, 'file': path.name, 'original_functions_preserved': list(original_functions), 'original_sha256': EXPECTED_SHA, 'derived_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'gpu_execution_verified': False}


if __name__ == '__main__':
    result = [build(mode) for mode in ['notebook', 'cosine']]
    (ROOT / 'notebook_source_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
