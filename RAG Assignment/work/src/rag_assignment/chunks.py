from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import os
import uuid
from pathlib import Path

from .common import fingerprint, read_json, sha256_file, write_json


def legal_sections(text: str) -> list[tuple[str, str]]:
    """Keep article boundaries; do not infer article labels from answer annotations."""
    pattern = re.compile(r"(?im)^\s*(Điều\s+\d+[a-zđ]?\s*[.:][^\n]*)")
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]
    parts = []
    if matches[0].start() > 0:
        parts.append(("Phần mở đầu", text[:matches[0].start()]))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((match.group(1).strip(), text[match.start():end]))
    return parts


def build_chunks(data_dir: Path, index_dir: Path, mode: str = "baseline", chunk_size: int | None = None,
                 overlap: int | None = None) -> dict:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if mode not in {"baseline", "legal"}:
        raise ValueError(mode)
    chunk_size = chunk_size or (2000 if mode == "baseline" else 1000)
    overlap = overlap if overlap is not None else (50 if mode == "baseline" else 100)
    params = {"mode": mode, "chunk_size": chunk_size, "overlap": overlap, "version": 1,
              "data_fingerprint": read_json(data_dir / "manifest.json")["fingerprint"]}
    identity = fingerprint(params)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "chunks_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != identity:
            raise ValueError("Chunk configuration changed; use a new index directory")
        if not (index_dir / "chunks.sqlite").exists():
            raise ValueError("Chunk manifest exists but DB is missing")
        return manifest
    db = index_dir / "chunks.sqlite"
    if db.exists():
        raise FileExistsError(f"Incomplete chunk DB exists: {db}; use a new directory")
    building_db = db.with_name(f"chunks.building-{uuid.uuid4().hex}.sqlite")
    conn = sqlite3.connect(building_db)
    conn.execute("CREATE TABLE chunks (row_id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT, source TEXT, law_id TEXT, section TEXT, text TEXT)")
    source = sqlite3.connect(data_dir / "corpus.sqlite")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
    start, count = time.perf_counter(), 0
    for doc_id, name, law_id, text in source.execute("SELECT doc_id,source,law_id,text FROM documents ORDER BY source"):
        if mode == "baseline":
            sections = [("", text.replace("\n", " ").lower().strip())]
        else:
            sections = legal_sections(text)
        local_index = 0
        for section, content in sections:
            for piece in splitter.split_text(content):
                if mode == "legal":
                    # Title repeats only corpus content; labels from RES are never used here.
                    prefix = f"Văn bản: {Path(name).stem}. {section}\n" if section else f"Văn bản: {Path(name).stem}\n"
                    piece = prefix + piece
                chunk_id = hashlib.sha256(f"{identity}:{doc_id}:{local_index}".encode()).hexdigest()[:24]
                conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", (count, chunk_id, doc_id, name, law_id, section, piece))
                count += 1
                local_index += 1
        if count and count % 20000 < 20:
            conn.commit()
    conn.commit()
    source.close()
    conn.close()
    os.replace(building_db, db)
    manifest = {**params, "fingerprint": identity, "chunks": count, "build_seconds": time.perf_counter() - start,
                "chunks_db_sha256": sha256_file(db)}
    write_json(manifest_path, manifest)
    return manifest


class ChunkStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = read_json(directory / "chunks_manifest.json")
        self.conn = sqlite3.connect(directory / "chunks.sqlite")
        self.conn.row_factory = sqlite3.Row

    def get(self, row_ids: list[int]) -> list[dict]:
        # k is small; independent queries preserve ranking and do not hit SQLite variable limits.
        return [dict(self.conn.execute("SELECT * FROM chunks WHERE row_id=?", (int(i),)).fetchone()) for i in row_ids]

    def texts(self):
        for row in self.conn.execute("SELECT text FROM chunks ORDER BY row_id"):
            yield row[0]

    def __len__(self):
        return self.manifest["chunks"]

    def close(self):
        self.conn.close()
