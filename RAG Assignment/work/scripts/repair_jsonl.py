"""Repair only an incomplete final JSONL record, retaining an exact backup."""
import argparse
import json
from pathlib import Path


def repair(path: Path) -> dict:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    invalid = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(i)
    if not invalid:
        return {"status": "unchanged", "path": str(path)}
    if invalid != [len(lines) - 1]:
        raise ValueError("Invalid record is not solely the final line. Manual investigation required; no edits made")
    backup = path.with_name(path.name + ".incomplete.bak")
    if backup.exists():
        raise FileExistsError(f"Backup exists; refusing to overwrite {backup}")
    backup.write_bytes(raw)
    path.write_bytes(b"".join(lines[:-1]))
    result = {"status": "removed_incomplete_final_line", "path": str(path), "backup": str(backup),
              "removed_line": len(lines), "removed_bytes": len(lines[-1])}
    with path.with_name(path.name + ".repair_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.path.resolve()), ensure_ascii=False, indent=2))
