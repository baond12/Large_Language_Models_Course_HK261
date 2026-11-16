"""Freeze all question IDs in the supplied RES workbook, in Excel row order."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import openpyxl

CODE = Path(__file__).resolve().parent
RES = CODE.parents[2] / 'RES.xlsx'
EXPECTED_SHA = 'd45e399006360bec31f9a73d7dc306263b9b2f580b4daab7aa2d017115d32750'


def main():
    digest = hashlib.sha256(RES.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError('Workbook differs from the supplied RES snapshot')
    wb = openpyxl.load_workbook(RES, read_only=True, data_only=True)
    try:
        rows = wb.worksheets[0].iter_rows(values_only=True)
        header = next(rows)
        qi, ai = header.index('Câu Hỏi'), header.index('Trả lời')
        ids = []
        for excel_row, row in enumerate(rows, start=2):
            if not any(value is not None for value in row):
                continue
            question, answer = str(row[qi] or '').strip(), str(row[ai] or '').strip()
            if not question or not answer:
                raise ValueError(f'Missing question/reference at Excel row {excel_row}')
            key = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', question)).casefold()
            ids.append('q_' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:16])
    finally:
        wb.close()
    if len(ids) != 2745 or len(set(ids)) != len(ids):
        raise ValueError('Expected 2745 unique pairs in the reviewed RES snapshot')
    manifest = {'selection': 'full-res', 'source_res_sha256': digest, 'count': len(ids), 'order': 'original Excel row order, starting after header', 'ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('utf-8')).hexdigest(), 'evaluation_history': 'Includes questions seen in prior development and pilot runs; not an unseen holdout.', 'ids': ids}
    (CODE / 'full_res_selection.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in manifest.items() if key != 'ids'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
