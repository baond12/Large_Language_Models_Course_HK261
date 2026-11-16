"""Read the frozen evaluation IDs independently of model loading."""
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent


def load_selection(name, code_dir=CODE):
    code_dir = Path(code_dir)
    if name == 'fixed-100':
        payload = json.loads((code_dir.parent / 'experiment_contract.json').read_text(encoding='utf-8'))
        ids = payload['planned_evaluation_ids']
    elif name == 'full-res':
        payload = json.loads((code_dir / 'full_res_selection.json').read_text(encoding='utf-8'))
        ids = payload['ids']
        digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('utf-8')).hexdigest()
        if len(ids) != payload['count'] or digest != payload['ids_sha256']:
            raise ValueError('Full RES selection count or ID digest differs from its manifest')
        if payload['source_res_sha256'] != 'd45e399006360bec31f9a73d7dc306263b9b2f580b4daab7aa2d017115d32750':
            raise ValueError('Full RES selection belongs to another workbook')
    else:
        raise ValueError(f'Unknown evaluation selection: {name}')
    if len(ids) < 100 or len(set(ids)) != len(ids):
        raise ValueError('A frozen evaluation selection must have at least 100 unique question IDs')
    return list(ids)


def resolve_selection(name, limit=None, code_dir=CODE):
    ids = load_selection(name, code_dir)
    count = len(ids) if limit is None else int(limit)
    if not 1 <= count <= len(ids):
        raise ValueError(f'--limit must be between 1 and {len(ids)} for {name}')
    return ids, ids[:count], count < len(ids)
