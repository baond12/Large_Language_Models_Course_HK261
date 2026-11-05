"""Deterministically build the teaching/demo notebook without executing GPU cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text):
    result = {"cell_type": kind, "id": f"rag-{len(cells):02d}", "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    cells.append(result)


cell("markdown", """# RAG Assignment Law Chatbot

Notebook hướng dẫn khám phá dữ liệu, chạy thực nghiệm và demo. Code chi tiết nằm trong `src/rag_assignment` để kiểm thử và tái sử dụng. Xem `SERVER_RUN.md` để cài dependency trong Docker P100 trước khi chạy. Notebook chưa được thực thi trên P100; output sẽ do phiên chạy thực tế của bạn tạo ra.

Mục tiêu: baseline công bằng, cải thiện có số liệu, giải thích được nguyên nhân. Không yêu cầu fine-tuning ngay từ đầu.
""")
cell("markdown", """## 1 Đường dẫn và tiện ích chạy

Khởi động Jupyter tại thư mục có `run.py`. Đặt SOURCE_ROOT thành thư mục chứa bốn tệp đầu vào. Không dùng đường dẫn Google Drive cứng trong code.
""")
cell("code", """from pathlib import Path
import json
import subprocess
import sys

PROJECT_ROOT = Path.cwd().resolve()
SOURCE_ROOT = PROJECT_ROOT.parent  # sửa thành /workspace/rag-inputs nếu cần
assert (PROJECT_ROOT / 'run.py').exists(), 'Mở notebook tại thư mục code'

def run(*args):
    subprocess.run([sys.executable, '-u', str(PROJECT_ROOT / 'run.py'), *map(str, args)], cwd=PROJECT_ROOT, check=True)

def read(relative):
    return json.loads((PROJECT_ROOT / relative).read_text(encoding='utf-8'))
""")
cell("markdown", """## 2 Kiểm tra môi trường và dữ liệu

Preflight phải chạy được FP16 trên GPU. P100 cần PyTorch có kernel Pascal. `prepare` lưu hash, schema, split theo nhóm văn bản và các tập diagnostic riêng.
""")
cell("code", """run('doctor', '--require-cuda')
run('prepare', '--source-root', SOURCE_ROOT)
audit = read('data/audit.json')
print(json.dumps({k: audit[k] for k in ['documents', 'questions', 'split_counts', 'reference_coverage']}, ensure_ascii=False, indent=2))
""")
cell("markdown", """## 3 Hiểu pipeline và ablation

- B0: chunk 2000/50, SBERT, L2 mặc định theo code baseline.
- B1: cùng chunk, cosine theo mô tả đề.
- R1: BM25 + dense cosine, RRF top-30 → top-5, cùng prompt gốc.
- R2: như R1, thay context packing và prompt có nguồn.
- R3 tùy chọn: chia văn bản theo Điều; cần bật trước khi xem test.

Giữ cùng Qwen1.5-1.8B và ngân sách token để đo đóng góp pipeline. Bản baseline gốc được giữ nguyên. Metadata nguồn không lấy từ nhãn câu hỏi.
""")
cell("markdown", """## 4 Pilot 3 câu trên mỗi cấu hình

Pilot kiểm tra vận hành, không thay benchmark 100 câu. Bước này vẫn xây dense index toàn corpus để không thay đổi không gian truy hồi. Kiểm tra predictions, nguồn và lỗi trước khi chạy đầy đủ.
""")
cell("code", """subprocess.run([sys.executable, '-u', str(PROJECT_ROOT / 'run_all.py'), '--stage', 'pilot', '--device', 'cuda', '--source-root', str(SOURCE_ROOT)], cwd=PROJECT_ROOT, check=True)
""")
cell("markdown", """## 5 Chạy đầy đủ và chọn bằng dev

Lệnh dưới chạy dev → lưu lựa chọn → test đủ 100 câu → chấm 5 metric → so sánh ghép cặp. Chạy lại cùng lệnh để tiếp tục. Không chỉnh phương pháp theo kết quả test đã xem rồi coi đó vẫn là test độc lập.
""")
cell("code", """subprocess.run([sys.executable, '-u', str(PROJECT_ROOT / 'run_all.py'), '--stage', 'full', '--device', 'cuda', '--source-root', str(SOURCE_ROOT)], cwd=PROJECT_ROOT, check=True)
selection = read('artifacts/selection.json')
print(selection['selected_config'])
""")
cell("markdown", """## 6 Xem kết quả và lỗi

Đối chiếu cả tăng và giảm; cosine không phải tính đúng pháp lý. BLEU trong evaluator là unigram đơn giản của baseline. Các ca lỗi vẫn nằm trong mẫu số. Primary test và diagnostic test có mục đích khác nhau.
""")
cell("code", """for path in sorted((PROJECT_ROOT / 'artifacts/comparisons').glob('*/comparison.json')):
    result = json.loads(path.read_text(encoding='utf-8'))
    print(path.parent.name, result['paired_questions'])
    print(json.dumps(result['deltas'], ensure_ascii=False, indent=2))
""")
cell("markdown", """## 7 Demo câu hỏi mới

Chỉ truyền câu hỏi vào inference. Thay ví dụ bằng câu cụ thể đã kiểm chứng có bằng chứng trong corpus trước buổi demo. Kiểm tra lại nội dung các nguồn; nhãn [S#] đúng cú pháp không đảm bảo lập luận đúng.
""")
cell("code", """selected_name = read('artifacts/selection.json')['selected_config']['name']
question = 'Quy định được trích trong nguồn áp dụng cho đối tượng nào và có các điều kiện gì?'
run('demo', '--config', f'configs/{selected_name}.json', '--device', 'cuda', '--question', question)
""")
cell("markdown", """## 8 Gói kết quả gửi lại

Gửi ZIP dưới đây cùng `p100_full.log`. Xem `REPORT.md`, `DEMO.md`, `DEFENSE.md` để hoàn thiện phần phân tích và bảo vệ bài sau khi có kết quả.
""")
cell("code", """run('report')
run('package', '--kind', 'results')
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "RAG_Assignment.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Built {len(cells)} cells")
