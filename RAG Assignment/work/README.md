# RAG Assignment Law Chatbot

Code thực nghiệm cho assignment HCMUT: baseline được sửa lỗi vận hành, hybrid retrieval, context có nguồn, đánh giá cùng tập câu hỏi và báo cáo từ kết quả thật.

**Trạng thái:** đã chuẩn bị dữ liệu và code; xem `artifacts/checklist.json` và `REPORT.md` để biết phép đo nào thực sự đã chạy. Chưa có kết quả generation trên P100 thì chưa thể tuyên bố cải thiện hoặc đủ điều kiện nộp.

## Bắt đầu trên server P100

Hướng dẫn chi tiết: **[SERVER_RUN.md](SERVER_RUN.md)**. Trong container có GPU, dùng Python 3.11, cài dependency và chạy pilot trước:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-core.txt -r requirements-models.txt
python run.py doctor --require-cuda
python run_all.py --stage pilot --device cuda --source-root /path/to/RAG_Assignment_inputs
python run_all.py --stage full --device cuda --source-root /path/to/RAG_Assignment_inputs
```

Thư mục `--source-root` phải chứa đúng bốn tệp: `Dataset.zip`, `RES.xlsx`, `Baseline.ipynb`, `RAG_Assignment.pdf`. Khi giữ cấu trúc môn học hiện tại, mặc định là thư mục cha của `work/`. Code đọc ZIP trực tiếp; không cần giải nén hoặc mount Google Drive. Đường dẫn có khoảng trắng phải đặt trong dấu nháy của shell.

## Các cấu hình

| Cấu hình | Thành phần thay đổi |
|---|---|
| b0_code | Baseline code: chunk 2000/50, SBERT thô, FAISS L2, top-5, prompt gốc |
| b1_cosine | Chuẩn hóa vector và inner product tương đương cosine theo mô tả đề |
| r1_hybrid | Kết hợp BM25 và dense cosine bằng RRF; giữ prompt gốc |
| r2_grounded | Như R1; chia ngân sách context cho nguồn và prompt yêu cầu dẫn nguồn |
| r3_legal | Tùy chọn: chia theo Điều, chunk 1000 ký tự/overlap 100 |

Mọi cấu hình chính giữ cùng `Qwen/Qwen1.5-1.8B`, greedy, 4096 input token và 256 generated token. Không mặc định fine-tune. R3 cần thêm thời gian/index, bật bằng `--include-legal` từ vòng dev đầu tiên; nếu đã xem test, không thêm cấu hình rồi tuyên bố test cũ vẫn là holdout độc lập.

## Quy trình và đầu ra

1. `prepare`: hash dữ liệu, audit corpus/RES, cố định split theo nhóm số hiệu văn bản.
2. `chunks`, `bm25`, `dense`: xây index; embedding ghi checkpoint để resume.
3. `generate --split dev`, `score`: chạy cùng ID, không đọc nhãn trong inference.
4. `select`: chọn theo dev ROUGE-L, tie-break bằng latency và tên.
5. `generate --split test`, `score`, `compare`: so sánh trên cùng >=100 câu, bootstrap ghép cặp.
6. `report`, `package`: tổng hợp báo cáo và ZIP kết quả để gửi lại.

`run_all.py` tự chạy các bước trên trong tiến trình riêng để giải phóng VRAM. Mẫu lỗi được lưu và chấm 0, không bỏ khỏi mẫu số. Nếu toàn bộ generation lỗi, xem `error` trong predictions trước khi diễn giải metric.

Các thư mục:

- `src/rag_assignment/`: data, chunking, retrieval, generation, evaluator, CLI và báo cáo.
- `configs/`: cấu hình ablation.
- `tests/`: metric kiểm tra tay, so với hàm trong notebook gốc, split, BM25, context và mô hình tiny cục bộ.
- `data/`: corpus SQLite, câu hỏi và nhãn tách riêng, splits, audit, hash nguồn.
- `indexes/`: chunk SQLite, BM25, vector memmap; không đưa vào Git.
- `runs/dev/`, `runs/test/`, `runs/pilot/`: manifest, predictions, scores và summary.
- `artifacts/`: môi trường, retrieval diagnostics, selection, comparisons và checklist.
- `dist/rag_assignment_results.zip`: tệp gửi lại để phân tích kết quả.

## Chạy local phần nhẹ

```powershell
.\.venv\Scripts\python.exe run_all.py --stage prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Môi trường local `.venv` là môi trường đã tạo trong phiên này. Khi clone sang máy khác cần tạo môi trường và cài requirements. Các test mô hình sẽ được skip nếu chưa cài dependency mô hình; `doctor` và log test ghi rõ điều đó.

## Đánh giá và giới hạn dữ liệu

Test chính có 100 câu, dev chính 107 câu với seed 261 ở dữ liệu hiện tại. Chỉ 51/2745 câu có ít nhất một số hiệu tham chiếu khớp tên tệp corpus. Theo split đã khai báo, dev/test chính không có khớp tên tệp; các tập diagnostic riêng gồm 29 và 22 câu có khớp. Đây là giới hạn quan trọng, không được đổi test chính thành tập dễ hơn mà không công bố.

Coverage tên tệp không đồng nghĩa answerability: văn bản khác có thể chứa nội dung liên quan, và số hiệu địa phương có thể không duy nhất. Không diễn giải filename-level retrieval metrics thành tính đúng pháp lý. Cần kiểm tra thủ công nội dung của demo và các ca tăng/giảm điểm.

Evaluator giữ cách tính lexical trong baseline và xuất thêm phiên bản tập hợp của đề. Cosine dùng cùng Vietnamese-SBERT revision cho mọi hệ thống. Encoder giới hạn 256 token, nên cosine không phản ánh toàn bộ đáp án dài. Chi tiết ở `docs/baseline_audit.md` và `docs/metrics.md`.

## Resume và thay đổi cấu hình

Chạy lại đúng lệnh để tiếp tục các câu chưa được ghi. Model revision, data hash và config phải khớp manifest; thay đổi thì tạo tên cấu hình/thư mục run mới. Mẫu đã lỗi vẫn được giữ để công bố; muốn retry phải tạo run mới cho cùng tập ID. Nếu tiến trình bị kill giữa lúc ghi JSONL, làm theo hướng dẫn phục hồi trong `SERVER_RUN.md`; không tự xóa mẫu lỗi.

Không đưa `.venv`, `.cache`, `data`, `indexes`, `runs` vào Git. Source ZIP không chứa mô hình hoặc dữ liệu gốc. Không có thao tác tự push, nộp bài hoặc sử dụng API trả phí.
