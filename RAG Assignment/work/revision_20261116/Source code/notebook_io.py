"""Input/output and provenance only; retrieval and generation stay in the notebook."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import unicodedata
import zipfile
from evaluation_selection import resolve_selection

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parent
CONTRACT = json.loads((ROOT / 'experiment_contract.json').read_text(encoding='utf-8'))
MODE = os.environ.get('BASELINE_DISTANCE', 'notebook')
if MODE not in {'notebook', 'cosine'}:
    raise ValueError('BASELINE_DISTANCE must be notebook or cosine')
RUN = Path(os.environ.get('BASELINE_RUN_DIR', WORK / 'runs/notebook_baseline_20261116' / MODE)).resolve()
DATA_ROOT = Path(os.environ.get('BASELINE_DATA_DIR', WORK.parent)).resolve()
CORPUS_ROOT = Path(os.environ.get('BASELINE_CORPUS_DIR', WORK / 'data/notebook_corpus_20261116')).resolve()
RES_CSV = RUN / 'evaluation_res.csv'
INDEX_DIR = RUN / 'faiss_db'
LOCK = json.loads((Path(__file__).resolve().parent / 'models.lock.json').read_text(encoding='utf-8'))
EMBEDDING_REVISION = LOCK['keepitreal/vietnamese-sbert']['revision']
LLM_REVISION = LOCK['Qwen/Qwen1.5-1.8B']['revision']
SELECTION_NAME = os.environ.get('BASELINE_EVALUATION_SET', 'fixed-100')
SELECTION_IDS, EVALUATION_IDS, IS_PILOT = resolve_selection(SELECTION_NAME, os.environ.get('BASELINE_QUESTION_LIMIT'))
N = len(EVALUATION_IDS)
_last = {}
_started = time.perf_counter()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def begin():
    import numpy as np
    import torch
    expected_python = os.environ.get('BASELINE_EXPECTED_PYTHON')
    if expected_python and Path(sys.executable).resolve() != Path(expected_python).resolve():
        raise RuntimeError(f'Jupyter kernel Python differs from checked Python: {sys.executable} != {expected_python}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=3 before starting this notebook kernel')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this run requires the user-operated GPU server')
    if len(CONTRACT['planned_evaluation_ids']) < 100 or len(set(CONTRACT['planned_evaluation_ids'])) != len(CONTRACT['planned_evaluation_ids']):
        raise ValueError('The full evaluation selection must contain at least 100 unique IDs')
    RUN.mkdir(parents=True, exist_ok=False)
    random.seed(261); np.random.seed(261); torch.manual_seed(261); torch.cuda.manual_seed_all(261)
    packages = ['torch', 'transformers', 'sentence-transformers', 'langchain-community', 'langchain-core', 'langchain-huggingface', 'langchain-text-splitters', 'faiss-cpu', 'numpy', 'pandas', 'openpyxl', 'scikit-learn', 'nbclient', 'nbformat', 'ipykernel']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if sha(DATA_ROOT / 'Baseline.ipynb') != CONTRACT['baseline_original_sha256']:
        raise ValueError('Original instructor notebook does not match the reviewed source')
    sources = {n: sha(DATA_ROOT / n) for n in ['Baseline.ipynb', 'RES.xlsx', 'Dataset.zip']}
    expected_inputs = {'Baseline.ipynb': CONTRACT['baseline_original_sha256'], 'RES.xlsx': 'd45e399006360bec31f9a73d7dc306263b9b2f580b4daab7aa2d017115d32750', 'Dataset.zip': 'd69d5a7229cc2a3b81e8dfe3c0d1feb295994daa58a982feac88244dc8244bca'}
    if sources != expected_inputs:
        raise ValueError('Supplied source input hashes differ from the reviewed original files')
    write_json(RUN / 'environment.json', {'python': sys.version, 'python_executable': sys.executable, 'packages': versions, 'source_sha256': sources, 'seed': 261, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'logical_device': 0, 'gpu': torch.cuda.get_device_name(0), 'distance_branch': MODE, 'evaluation_count': N, 'limited_run': IS_PILOT, 'model_revisions': LOCK, 'index_logic_sha256': os.environ.get('BASELINE_INDEX_LOGIC_SHA256'), 'baseline_logic': 'instructor notebook cells with documented operational edits'})
    environment = json.loads((RUN / 'environment.json').read_text(encoding='utf-8'))
    environment.update({'evaluation_set': SELECTION_NAME, 'expected_selection_count': len(SELECTION_IDS)})
    write_json(RUN / 'environment.json', environment)
    print(json.dumps({'run': str(RUN), 'questions': N, 'gpu': torch.cuda.get_device_name(0), 'physical_gpu_selection': '3', 'distance_branch': MODE}))


def prepare_evaluation():
    import pandas as pd
    frame = pd.read_excel(DATA_ROOT / 'RES.xlsx', sheet_name=0, dtype=object).fillna('')
    required = ['Câu Hỏi', 'Trả lời']
    if any(c not in frame.columns for c in required):
        raise ValueError('RES must contain question and reference columns')
    by_id = {}
    for i, row in frame.iterrows():
        question, answer = str(row['Câu Hỏi']).strip(), str(row['Trả lời']).strip()
        if not question and not answer:
            continue
        if not question or not answer:
            raise ValueError(f'Missing question or reference at Excel row {i+2}')
        key = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', question)).casefold()
        uid = 'q_' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
        if uid in by_id:
            raise ValueError(f'Duplicate question identity: {uid}')
        by_id[uid] = {'question_id': uid, 'excel_row': int(i)+2, 'Câu Hỏi': question, 'Trả lời': answer}
    selected = [by_id[uid] for uid in EVALUATION_IDS]
    pd.DataFrame(selected).to_csv(RES_CSV, index=False, encoding='utf-8')
    write_json(RUN / 'evaluation_pairs.json', selected)
    return selected


def prepare_inputs():
    prepare_evaluation()
    zip_path = DATA_ROOT / 'Dataset.zip'
    # Extract only provided text members. No path from the ZIP may escape the corpus root.
    archive_sha = sha(zip_path)
    marker = CORPUS_ROOT / '.archive_sha256'
    if marker.is_file():
        if marker.read_text(encoding='ascii').strip() != archive_sha:
            raise ValueError('Corpus extraction belongs to another ZIP; choose a new corpus directory')
    else:
        if CORPUS_ROOT.exists() and any(CORPUS_ROOT.iterdir()):
            raise ValueError('Refusing to reuse an unverified nonempty corpus directory')
        CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
        filenames = []
        with zipfile.ZipFile(zip_path) as archive:
            for item in archive.infolist():
                if item.is_dir() or not item.filename.endswith('.txt'):
                    continue
                # The original notebook reads the immediate export_1 directory.
                name = item.filename.replace('\\', '/')
                parts = Path(name).parts
                if '..' in parts or Path(name).is_absolute() or ':' in name:
                    raise ValueError(f'Unsafe archive member: {name}')
                if len(parts) < 2 or parts[-2] != 'export_1':
                    raise ValueError(f'Unexpected text member layout; inspect before changing the loader: {name}')
                target = CORPUS_ROOT / parts[-1]
                if target.name in filenames:
                    raise ValueError(f'Duplicate corpus basename: {target.name}')
                target.write_bytes(archive.read(item))
                filenames.append(target.name)
        if not filenames:
            raise ValueError('No export_1 text files found')
        marker.write_text(archive_sha + '\n', encoding='ascii')
    return str(CORPUS_ROOT)


def document_filenames(path):
    """Freeze the first filesystem order, without sorting the instructor's original loop."""
    path = Path(path)
    supplied = os.environ.get('BASELINE_DOCUMENT_ORDER')
    manifest = Path(supplied).resolve() if supplied else CORPUS_ROOT / '.document_order.json'
    current = [n for n in os.listdir(path) if n.endswith('.txt')]
    if manifest.is_file():
        names = json.loads(manifest.read_text(encoding='utf-8'))
        if set(names) != set(current) or len(names) != len(current):
            raise ValueError('Corpus file list differs from frozen order')
    else:
        if supplied:
            raise FileNotFoundError(f'Provided document order does not exist: {manifest}')
        names = current
        write_json(manifest, names)
    write_json(RUN / 'document_order.json', names)
    return names


def record_index(store, documents=None, chunks=None):
    import faiss
    import numpy as np
    index = store.index
    samples = np.stack([index.reconstruct(i) for i in range(min(16, index.ntotal))])
    info = {'index_type': type(index).__name__, 'index_metric_type': index.metric_type, 'faiss_metric_l2': faiss.METRIC_L2, 'faiss_metric_inner_product': faiss.METRIC_INNER_PRODUCT, 'distance_strategy': str(store.distance_strategy), 'normalize_L2': bool(getattr(store, '_normalize_L2', False)), 'vectors': index.ntotal, 'dimension': index.d, 'sample_vector_norms': np.linalg.norm(samples, axis=1).tolist(), 'documents': documents, 'chunks': chunks}
    if MODE == 'cosine' and (index.metric_type != faiss.METRIC_INNER_PRODUCT or not info['normalize_L2']):
        raise ValueError('Cosine branch was not built with normalized vectors and inner-product search')
    info['index_files_sha256'] = {name: sha(INDEX_DIR / name) for name in ['index.faiss', 'index.pkl']}
    write_json(RUN / 'index_info.json', info)
    print(json.dumps(info))


def reuse_index():
    """Reuse only an index produced by this instructor-notebook workflow."""
    global INDEX_DIR
    source = Path(os.environ['BASELINE_REUSE_INDEX_FROM']).resolve()
    old = json.loads((source / 'environment.json').read_text(encoding='utf-8'))
    current = json.loads((RUN / 'environment.json').read_text(encoding='utf-8'))
    for key in ['source_sha256', 'model_revisions', 'distance_branch', 'index_logic_sha256']:
        if old.get(key) != current.get(key) or old.get(key) is None:
            raise ValueError(f'Cannot reuse index: incompatible {key}')
    for package in ['torch', 'transformers', 'sentence-transformers', 'langchain-community', 'langchain-core', 'langchain-text-splitters', 'faiss-cpu', 'numpy']:
        if old.get('packages', {}).get(package) != current.get('packages', {}).get(package):
            raise ValueError(f'Cannot reuse index with changed dependency: {package}')
    info = json.loads((source / 'index_info.json').read_text(encoding='utf-8'))
    source_index = Path(info.get('reused_index_dir', source / 'faiss_db'))
    hashes = info.get('index_files_sha256', {})
    if set(hashes) != {'index.faiss', 'index.pkl'}:
        raise ValueError('Cannot reuse index without recorded index-file hashes')
    for name, expected in hashes.items():
        if sha(source_index / name) != expected:
            raise ValueError(f'Cannot reuse modified index file: {name}')
    INDEX_DIR = source_index
    info = dict(info, reused_from=str(source), reused_index_dir=str(source_index))
    write_json(RUN / 'index_info.json', info)
    write_json(RUN / 'document_order.json', json.loads((source / 'document_order.json').read_text(encoding='utf-8')))
    print(f'Reusing verified notebook index: {INDEX_DIR}', flush=True)


def record_model(llm, embeddings):
    model, tokenizer = llm.pipeline.model, llm.pipeline.tokenizer
    write_json(RUN / 'model_info.json', {'model': model.config._name_or_path, 'revision': getattr(model.config, '_commit_hash', None), 'parameters': sum(p.numel() for p in model.parameters()), 'dtype': str(model.dtype), 'generation_config': model.generation_config.to_dict(), 'pipeline_kwargs': getattr(llm, 'pipeline_kwargs', {}), 'model_kwargs': getattr(llm, 'model_kwargs', {}), 'tokenizer_model_max_length': tokenizer.model_max_length, 'embedding_revision_requested': EMBEDDING_REVISION, 'generation_call_pipeline_kwargs': {'max_new_tokens': 256, 'return_full_text': False}, 'extra_input_token_budget': None, 'completion_slicing_added': False})
    if sum(p.numel() for p in model.parameters()) >= 4_000_000_000:
        raise ValueError('LLM must have fewer than 4 billion parameters')
    if not getattr(model, '_baseline_observer_installed', False):
        original_generate = model.generate
        def observed_generate(*args, **kwargs):
            output = original_generate(*args, **kwargs)
            config = kwargs.get('generation_config', model.generation_config)
            limit = kwargs.get('max_new_tokens', getattr(config, 'max_new_tokens', None))
            _last['backend_generation'] = {'effective_max_new_tokens': limit, 'do_sample': kwargs.get('do_sample', getattr(config, 'do_sample', None))}
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            sequences = getattr(output, 'sequences', output)
            if input_ids is not None and hasattr(input_ids, 'shape') and hasattr(sequences, 'shape'):
                _last['input_tokens'] = int(input_ids.shape[-1])
                _last['generated_tokens'] = int(sequences.shape[-1]) - int(input_ids.shape[-1])
                _last['reached_generation_cap'] = bool(limit is not None and _last['generated_tokens'] >= limit)
            return output
        model.generate = observed_generate
        model._baseline_observer_installed = True


def capture_context(question, docs, prompt):
    _last.clear()
    _last.update({'question': question, 'prompt': prompt, 'passages': [{'text': getattr(d, 'page_content', str(d)), 'metadata': getattr(d, 'metadata', {})} for d in docs], 'error': None, 'started': time.perf_counter()})


def capture_error(question):
    error = traceback.format_exc()
    if _last.get('question') != question:
        _last.clear()
    _last['error'] = error
    with (RUN / 'errors.log').open('a', encoding='utf-8') as f:
        f.write(question + '\n' + error + '\n')
    print(error)


def record_result(idx, row, response, metrics):
    item = {'id': row['question_id'], 'excel_row': int(row['excel_row']), 'question': row['Câu Hỏi'], 'reference': str(row['Trả lời']).strip(), 'answer': response, 'metrics_notebook': metrics, 'error': _last.get('error'), 'prompt': _last.get('prompt'), 'passages': _last.get('passages', []), 'returned_text_starts_with_prompt': bool(_last.get('prompt') and response.startswith(_last['prompt'])), 'latency_seconds': time.perf_counter()-_last['started'] if _last.get('started') else None}
    for key in ['backend_generation', 'input_tokens', 'generated_tokens', 'reached_generation_cap']:
        item[key] = _last.get(key)
    with (RUN / 'predictions.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'RECORDED {idx+1}/{N} {row["question_id"]} error={bool(item["error"])}', flush=True)
    _last.clear()


def finish(results_df, mean_metrics, normalize_text):
    import math
    records = [json.loads(line) for line in (RUN / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['id'] for r in records] != EVALUATION_IDS or len(results_df) != N:
        raise ValueError('Output IDs/order/count do not match the frozen evaluation selection')
    scores = []
    for record in records:
        pred, ref = normalize_text(record['answer']).split(), normalize_text(record['reference']).split()
        p, r = set(pred), set(ref)
        overlap = len(p & r)
        bp = 1 if len(pred) >= len(ref) else math.exp(1-len(ref)/len(pred)) if pred else 0
        scores.append({'id': record['id'], 'Cosine_Similarity': record['metrics_notebook']['Cosine_Similarity'], 'Jaccard_Similarity': record['metrics_notebook']['Jaccard_Similarity'], 'Token_Overlap_Set_PDF': overlap/len(r) if r else 0, 'BLEU_Set_PDF': bp*overlap/len(p) if p else 0, 'ROUGE_L': record['metrics_notebook']['ROUGE_L']})
    pdf_means = {k: sum(row[k] for row in scores)/N for k in scores[0] if k != 'id'}
    write_json(RUN / 'scores_pdf.json', scores)
    prompt_count = sum(r['returned_text_starts_with_prompt'] for r in records)
    contract_violations = sum(not r['error'] and (r.get('backend_generation') is None or r['backend_generation'].get('effective_max_new_tokens') != 256 or r.get('generated_tokens') is None or r['generated_tokens'] > 256) for r in records)
    summary = {'status': 'completed_with_errors' if any(r['error'] for r in records) else 'notebook_execution_complete', 'questions': N, 'evaluation_set': SELECTION_NAME, 'expected_selection_count': len(SELECTION_IDS), 'limited_run': IS_PILOT, 'eligible_for_final_evaluation': not IS_PILOT and N >= 100 and not any(r['error'] for r in records) and prompt_count == 0 and contract_violations == 0, 'errors': sum(bool(r['error']) for r in records), 'empty_answers': sum(not r['answer'].strip() for r in records), 'returned_prompt_prefix_count': prompt_count, 'generation_contract_violations': contract_violations, 'max_observed_generated_tokens': max((r['generated_tokens'] for r in records if isinstance(r.get('generated_tokens'), int)), default=None), 'metrics_notebook': mean_metrics, 'metrics_pdf': pdf_means, 'distance_branch': MODE, 'elapsed_seconds': time.perf_counter()-_started, 'predictions_sha256': sha(RUN / 'predictions.jsonl'), 'rerun_independently_verified': False}
    write_json(RUN / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
