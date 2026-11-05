"""Record validation evidence. Tiny model tests do not claim task performance."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_assignment.common import implementation_identity, runtime_info, write_json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTHONIOENCODING"] = "utf-8"

for path in [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in (ROOT / "configs").glob("*.json"):
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["max_input_tokens"] > 0 and config["max_new_tokens"] > 0
for path in ROOT.glob("*.ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                        capture_output=True, text=True, encoding="utf-8")
log = result.stdout + result.stderr
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "artifacts/tests.log").write_text(log, encoding="utf-8")
match = re.search(r"Ran (\d+) tests", log)
skipped = re.search(r"OK \(skipped=(\d+)\)", log)
evidence = {"exit_code": result.returncode, "tests": int(match.group(1)) if match else None,
            "skipped": int(skipped.group(1)) if skipped else 0, "runtime": runtime_info(),
            "implementation": implementation_identity(ROOT),
            "scope": "unit/integration checks including tiny random local weights; not a real-model RAG benchmark"}
write_json(ROOT / "artifacts/validation.json", evidence)
print(log)
sys.exit(result.returncode)
