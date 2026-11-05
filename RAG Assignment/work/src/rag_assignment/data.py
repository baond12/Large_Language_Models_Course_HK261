from __future__ import annotations

import hashlib
import re
import sqlite3
import statistics
import unicodedata
import zipfile
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from .common import fingerprint, read_json, sha256_file, write_json, write_jsonl

LAW_RE = re.compile(r"\b\d+[A-Za-zĐđ]*(?:[/_]\d{4})?[/_][A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ0-9]*(?:[-_][A-Za-zÀ-ỹĐđ0-9]+)*", re.UNICODE)


def canonical_law_id(value: str) -> str:
    """Canonicalize legal identifiers without consulting any question labels."""
    value = unicodedata.normalize("NFC", value).upper().strip().replace("Đ", "D")
    value = value.replace("_", "/")
    value = re.sub(r"[^A-Z0-9/]+", "", value)
    return value.replace("/", "")


def law_ids_in_text(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s*([/\-])\s*", r"\1", text)
    return sorted({canonical_law_id(m.group()) for m in LAW_RE.finditer(text)})


def filename_law_id(name: str) -> str:
    stem = Path(name).stem
    # Corpus suffix _1/_2 disambiguates duplicate filenames, not the legal number.
    stem = re.sub(r"_\d+$", "", stem)
    return canonical_law_id(stem)


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"count": len(values), "min": ordered[0], "median": statistics.median(ordered),
            "p90": ordered[int((len(ordered) - 1) * .90)],
            "p95": ordered[int((len(ordered) - 1) * .95)], "max": ordered[-1],
            "mean": statistics.mean(ordered)}


def read_res(path: Path) -> list[dict]:
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    iterator = sheet.iter_rows(values_only=True)
    headers = next(iterator)
    required = ["Câu Hỏi", "Trả lời", "Số hiệu VBPL (Trích xuất)"]
    positions = {name: headers.index(name) for name in required}
    records = []
    seen = set()
    for excel_row, row in enumerate(iterator, start=2):
        if not any(v is not None for v in row):
            continue
        q, a, law = [str(row[positions[k]] or "").strip() for k in required]
        if not q or not a:
            raise ValueError(f"Missing question/answer in RES row {excel_row}; review rather than silently drop")
        qkey = unicodedata.normalize("NFC", re.sub(r"\s+", " ", q)).casefold()
        uid = "q_" + hashlib.sha256(qkey.encode()).hexdigest()[:16]
        if uid in seen:
            raise ValueError(f"Duplicate normalized question at RES row {excel_row}; review before split")
        seen.add(uid)
        ids = law_ids_in_text(law)
        # Free-text markers such as 'Không trích xuất được' are unknown labels,
        # never one shared legal-document group.
        records.append({"id": uid, "question": q, "answer": a, "law_ids": ids,
                        "law_label_raw": law, "excel_row": excel_row})
    workbook.close()
    return records


def make_splits(records: list[dict], seed: int = 261, test_size: int = 100, dev_size: int = 100,
                corpus_ids: set[str] | None = None) -> dict:
    """Group questions sharing any labeled legal document (connected components)."""
    import random
    if test_size <= 0 or dev_size <= 0:
        raise ValueError("Requested dev/test sizes must be positive")
    parents = list(range(len(records)))

    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, record in enumerate(records):
        for law in record["law_ids"]:
            if law in owners:
                parents[find(i)] = find(owners[law])
            else:
                owners[law] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record["id"])
    components = sorted(groups.values(), key=lambda group: sorted(group)[0])
    random.Random(seed).shuffle(components)
    splits = {"test": [], "dev": [], "unused": []}
    for group in components:
        split = "test" if len(splits["test"]) < test_size else "dev" if len(splits["dev"]) < dev_size else "unused"
        splits[split].extend(sorted(group))
    if len(splits["test"]) < test_size or len(splits["dev"]) < dev_size:
        raise ValueError("Not enough independent legal-document groups for requested splits")
    if corpus_ids is not None:
        # Separate corpus-covered diagnostics. Do not replace the primary random holdout
        # with an easier set, and never tune using diagnostic_test.
        lookup = {r["id"]: r for r in records}
        unused = set(splits["unused"])
        candidates = [g for g in components if set(g).issubset(unused)
                      and any(set(lookup[q]["law_ids"]) & corpus_ids for q in g)]
        splits["diagnostic_dev"], splits["diagnostic_test"] = [], []
        for group in candidates:
            target = "diagnostic_dev" if len(splits["diagnostic_dev"]) <= len(splits["diagnostic_test"]) else "diagnostic_test"
            splits[target].extend(group)
        diagnostic_ids = set(splits["diagnostic_dev"]) | set(splits["diagnostic_test"])
        splits["unused"] = [q for q in splits["unused"] if q not in diagnostic_ids]
    return {"seed": seed, "method": "legal_document_connected_components_v2", "requested_test_size": test_size,
            "requested_dev_size": dev_size, "ids": splits, "counts": {k: len(v) for k, v in splits.items()}}


def prepare(source_dir: Path, output_dir: Path, seed: int = 261, test_size: int = 100, dev_size: int = 100) -> dict:
    """Read-only ZIP/XLSX inputs; write reproducible corpus DB and separate questions/labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = [source_dir / n for n in ["Dataset.zip", "RES.xlsx", "Baseline.ipynb", "RAG_Assignment.pdf"]]
    source_hashes = {p.name: sha256_file(p) for p in originals}
    identity = fingerprint({"sources": source_hashes, "seed": seed, "test_size": test_size, "dev_size": dev_size,
                            "schema_version": 4})
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["fingerprint"] != identity:
            raise ValueError("Data manifest changed. Use a new output directory; do not overwrite a frozen split")
        expected = [output_dir / n for n in ["corpus.sqlite", "questions.jsonl", "labels.jsonl", "splits.json", "audit.json"]]
        if not all(p.exists() for p in expected):
            raise ValueError("Incomplete prepared dataset; use a fresh output directory")
        return read_json(output_dir / "audit.json")
    records = read_res(source_dir / "RES.xlsx")
    db_path = output_dir / "corpus.sqlite"
    if db_path.exists():
        raise FileExistsError(f"Refusing to replace existing DB without a complete manifest: {db_path}")
    building_db = db_path.with_name(f"corpus.building-{uuid.uuid4().hex}.sqlite")
    conn = sqlite3.connect(building_db)
    conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source TEXT UNIQUE, law_id TEXT, text TEXT, sha256 TEXT)")
    corpus_ids = set()
    doc_lengths, decode_errors, type_counts, hashes, header_mismatches = [], [], Counter(), Counter(), []
    with zipfile.ZipFile(source_dir / "Dataset.zip") as archive:
        members = sorted((m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".txt")), key=lambda x: x.filename)
        for member in members:
            raw = archive.read(member)
            try:
                # Match TextLoader/open(..., encoding='utf-8') universal newline handling.
                # Preserve original code points/BOM; NFC would change baseline chunking.
                text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeDecodeError as e:
                decode_errors.append({"source": member.filename, "error": str(e)})
                continue
            digest = hashlib.sha256(raw).hexdigest()
            doc_id = "d_" + hashlib.sha256(member.filename.encode()).hexdigest()[:16]
            law_id = filename_law_id(member.filename)
            header = re.search(r"Số\s*:\s*([^\n\r]+)", text[:2500], flags=re.IGNORECASE)
            if header:
                header_ids = law_ids_in_text(header.group(1))
                if header_ids and law_id not in header_ids:
                    header_mismatches.append({"source": member.filename, "filename_law_id": law_id, "header_law_ids": header_ids})
            conn.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (doc_id, member.filename, law_id, text, digest))
            corpus_ids.add(law_id)
            hashes[digest] += 1
            doc_lengths.append(len(text))
            suffix = Path(member.filename).stem.split("_")
            type_counts[suffix[-1] if not suffix[-1].isdigit() else suffix[-2] if len(suffix) > 1 else "numeric_filename"] += 1
        conn.commit()
    conn.execute("CREATE INDEX documents_law_id ON documents(law_id)")
    conn.close()
    if decode_errors:
        write_json(output_dir / "decode_errors.json", decode_errors)
        raise ValueError("Corpus has decoding errors; inspect decode_errors.json before proceeding")
    os.replace(building_db, db_path)
    splits = make_splits(records, seed, test_size, dev_size, corpus_ids)
    coverage = {}
    by_id = {r["id"]: r for r in records}
    for split, ids in splits["ids"].items():
        subset = [by_id[i] for i in ids]
        coverage[split] = {"questions": len(subset),
            "any_reference_in_corpus": sum(bool(set(r["law_ids"]) & corpus_ids) for r in subset),
            "all_references_in_corpus": sum(bool(r["law_ids"]) and set(r["law_ids"]).issubset(corpus_ids) for r in subset)}
    law_counts = Counter(law for r in records for law in r["law_ids"])
    audit = {"schema_version": 4, "data_fingerprint": identity, "documents": len(doc_lengths),
        "unique_legal_ids_in_corpus": len(corpus_ids), "exact_duplicate_documents": sum(n - 1 for n in hashes.values()),
        "empty_documents": sum(n == 0 for n in doc_lengths), "utf8_errors": len(decode_errors),
        "document_characters": distribution(doc_lengths), "document_filename_types": dict(type_counts.most_common()),
        "filename_header_id_mismatches": len(header_mismatches), "filename_header_mismatch_examples": header_mismatches[:20],
        "questions": len(records), "question_characters": distribution([len(r["question"]) for r in records]),
        "answer_characters": distribution([len(r["answer"]) for r in records]),
        "unique_reference_law_ids": len(law_counts), "questions_without_parseable_law_label": sum(not r["law_ids"] for r in records),
        "reference_law_frequency_top20": [list(item) for item in law_counts.most_common(20)],
        "reference_ids_not_in_corpus": sorted(set(law_counts) - corpus_ids), "reference_coverage": coverage,
        "split_counts": splits["counts"],
        "label_limitation": "Filename-level legal-document labels are weak relevance labels, not passage ground truth or legal correctness labels. Header mismatches are audited, not silently repaired.",
        "original_files_sha256": source_hashes}
    write_jsonl(output_dir / "questions.jsonl", ({"id": r["id"], "question": r["question"]} for r in records))
    write_jsonl(output_dir / "labels.jsonl", ({k: v for k, v in r.items() if k != "question"} for r in records))
    write_json(output_dir / "splits.json", splits)
    write_json(output_dir / "audit.json", audit)
    write_json(manifest_path, {"fingerprint": identity, "original_files_sha256": source_hashes, "schema_version": 4})
    return audit
