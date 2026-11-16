"""Check the Docker environment or execute the derived instructor notebook with Jupyter."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]


def check():
    required = {'numpy': 'numpy', 'pandas': 'pandas', 'openpyxl': 'openpyxl', 'scikit-learn': 'sklearn', 'tqdm': 'tqdm', 'torch': 'torch', 'transformers': 'transformers', 'sentence-transformers': 'sentence_transformers', 'faiss-cpu': 'faiss', 'langchain-community': 'langchain_community', 'langchain-text-splitters': 'langchain_text_splitters', 'langchain-huggingface': 'langchain_huggingface', 'nbclient': 'nbclient', 'nbformat': 'nbformat', 'ipykernel': 'ipykernel'}
    versions = {}
    missing = []
    for package, module in required.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        try:
            versions[package] = importlib.metadata.version(package) if present else None
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'importable_metadata_unavailable' if present else None
        if not present:
            missing.append(package)
    info = {'python': sys.version, 'platform': platform.platform(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'packages': versions, 'missing_packages': missing, 'files': {n: (WORK.parent/n).is_file() for n in ['Dataset.zip', 'RES.xlsx', 'Baseline.ipynb']}, 'notebook_execution_attempted': False}
    errors = []
    if not missing:
        # Import the exact notebook integrations before starting a long indexing job.
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_huggingface import HuggingFacePipeline
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception:
            errors.append(traceback.format_exc())
    if 'torch' not in missing:
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            info['logical_gpu_count'] = torch.cuda.device_count()
            if info['cuda_available']:
                info['gpu'] = torch.cuda.get_device_name(0)
                info['cuda_build'] = torch.version.cuda
        except Exception:
            errors.append(traceback.format_exc())
    info['import_errors'] = errors
    info['preflight_ready'] = not missing and not errors and all(info['files'].values()) and info.get('cuda_available', False) and info['CUDA_VISIBLE_DEVICES'] == '3'
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info['preflight_ready'] else 2


def execute(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise SystemExit('Set CUDA_VISIBLE_DEVICES=3 before launching the Jupyter kernel')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_name):
        raise SystemExit('Run name must be a simple new directory name')
    from nbclient import NotebookClient
    import nbformat
    from prepare_notebooks import build
    build(args.mode)
    notebook_path = HERE / f'Baseline_Docker_{args.mode}.ipynb'
    artifact = WORK / 'artifacts/notebook_execution_20261116' / args.run_name
    run_dir = WORK / 'runs/notebook_baseline_20261116' / args.run_name
    if artifact.exists() or run_dir.exists():
        raise SystemExit('Use a new run name; existing logs/results are preserved')
    artifact.mkdir(parents=True)
    os.environ['BASELINE_RUN_DIR'] = str(run_dir)
    os.environ['BASELINE_DISTANCE'] = args.mode
    os.environ['BASELINE_EXPECTED_PYTHON'] = sys.executable
    full_size = len(json.loads((HERE.parent / 'experiment_contract.json').read_text(encoding='utf-8'))['planned_evaluation_ids'])
    limit = args.limit if args.limit is not None else full_size
    if not 1 <= limit <= full_size:
        raise SystemExit(f'--limit must be between 1 and {full_size}')
    os.environ['BASELINE_QUESTION_LIMIT'] = str(limit)
    if args.reuse_index_from:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.reuse_index_from):
            raise SystemExit('--reuse-index-from must be a previous run name')
        os.environ['BASELINE_REUSE_INDEX_FROM'] = str(WORK / 'runs/notebook_baseline_20261116' / args.reuse_index_from)
    else:
        os.environ.pop('BASELINE_REUSE_INDEX_FROM', None)
    nb = nbformat.read(notebook_path, as_version=4)
    os.environ['BASELINE_INDEX_LOGIC_SHA256'] = hashlib.sha256(nb.cells[3].source.encode('utf-8')).hexdigest()
    events = []
    def completed(cell, cell_index, execute_reply):
        events.append({'cell': cell_index, 'status': execute_reply['content'].get('status'), 'utc': datetime.now(timezone.utc).isoformat()})
        nbformat.write(nb, artifact / 'executed.ipynb')
        print(f'CELL {cell_index}: {events[-1]["status"]}', flush=True)
    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg['header']['msg_type'] == 'stream':
                print(msg['content'].get('text', ''), end='', flush=True)
            return super().process_message(msg, cell, cell_index)
    client = StreamingNotebookClient(nb, timeout=None, kernel_name='python3', allow_errors=False, resources={'metadata': {'path': str(WORK)}}, on_cell_executed=completed)
    success = False
    try:
        client.execute()
        success = True
    except BaseException:
        (artifact / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        nbformat.write(nb, artifact / 'executed.ipynb')
        (artifact / 'execution_status.json').write_text(json.dumps({'execution_completed': success, 'events': events, 'run_dir': str(run_dir), 'notebook_source': str(notebook_path)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        log = []
        for i, cell in enumerate(nb.cells):
            log.append(f'\n===== CELL {i} =====\n')
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    log.append(output.get('text', ''))
                elif output.get('output_type') == 'error':
                    log.append('\n'.join(output.get('traceback', [])))
        (artifact / 'notebook.log').write_text(''.join(log), encoding='utf-8')
        with (artifact / 'pip-freeze.txt').open('w', encoding='utf-8') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        print(f'Execution evidence: {artifact}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=['notebook', 'cosine'], default='notebook')
    parser.add_argument('--run-name')
    parser.add_argument('--limit', type=int, help='Pilot question count; omit for the full fixed set of at least 100')
    parser.add_argument('--reuse-index-from', help='Previous run name whose instructor-notebook index will be verified and reused')
    args = parser.parse_args()
    if args.check:
        return check()
    if not args.run_name:
        parser.error('--execute requires --run-name')
    execute(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
