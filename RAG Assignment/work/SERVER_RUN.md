# Chạy trên P100 trong Docker

Người học chạy trên server riêng rồi gửi kết quả về. Đã nhận log preflight P100 và pilot từ server; không có kết nối điều khiển server trực tiếp từ phiên chat.

## Môi trường Docker hiện tại

Theo yêu cầu người học, chạy trực tiếp trong container `baond_llm`, chỉ cài dependency còn thiếu. Docker Python 3.10/PyTorch 2.2.1+cu118 đã vượt qua preflight và pilot base; không cần tạo venv hoặc tự động cài lại môi trường theo các lệnh tham khảo bên dưới. Mọi run lưu phiên bản thư viện thực tế.

GPU 1 hiện có dịch vụ `uvicorn` chiếm VRAM theo log người học; dùng `CUDA_VISIBLE_DEVICES=3` cho pilot tiếp theo. Lệnh có tải dữ liệu/model cần proxy `http://proxy.hcm.fpt.vn:80` qua `http_proxy` và `https_proxy`. Khi dùng `docker exec`, truyền các biến bằng `-e` vì export ở một shell khác không tự được giữ cho exec mới.

Đã có corpus, split, BM25 và 176.506 vector SBERT trên server. Cập nhật nhánh Instruct không cần xây lại các index này. Chỉ sau khi cập nhật code và kiểm thử, khóa thêm model bằng `lock-models --model Qwen/Qwen2.5-3B-Instruct` rồi chạy cấu hình mới với thư mục run mới. Xem `docs/instruct_experiment.md`.

## 1 Đưa code và dữ liệu lên server

Clone repository của bạn có thư mục này, hoặc giải nén `dist/rag_assignment_source.zip` vào thư mục làm việc. Source ZIP là gói code, chưa phải repository được tự động push lên GitHub.

Đặt bốn tệp đầu vào trong một thư mục riêng:

```text
/workspace/rag-inputs/
  Dataset.zip
  RES.xlsx
  Baseline.ipynb
  RAG_Assignment.pdf
/workspace/rag-work/
  run.py
  run_all.py
  requirements-core.txt
  requirements-models.txt
  src/
  configs/
  ...
```

Giữ `/workspace/rag-work` trên volume bền vững để không mất index/cache/run khi container ngừng. Dự trù nhiều GB cho model cache, SQLite, BM25 và vector; kiểm tra dung lượng trước khi chạy. Chưa có phép đo tổng dung lượng trên server.

## 2 Cài trong container hiện có

Nếu khởi tạo container mới, có thể tham khảo cấu hình Python 3.11 bên dưới. Đây không phải yêu cầu cài lại Docker đang chạy của người học. Nếu container đã có PyTorch hoạt động trên P100, kiểm tra `doctor` trước khi quyết định thay phiên bản:

```bash
cd /workspace/rag-work
python --version
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-core.txt -r requirements-models.txt
python run.py doctor --require-cuda
```

P100 là Pascal. Một số wheel CUDA mới không còn kernel Pascal; vì vậy không dùng `pip install -U torch` tùy tiện. `doctor` kiểm tra GPU, capability, danh sách kiến trúc và thực sự chạy FP16 matmul. Mô hình dùng FP16 và eager attention trên CUDA; không yêu cầu BF16, FlashAttention hoặc bitsandbytes.

Nếu dùng Dockerfile đi kèm, từ thư mục code:

```bash
docker build -t rag-assignment-p100 .
docker run --gpus all --ipc=host -it \
  -v /path/to/rag-work:/workspace/work \
  -v /path/to/rag-inputs:/workspace/inputs:ro \
  -w /workspace/work rag-assignment-p100 bash
```

Đây là lựa chọn bổ sung. Không cần tạo lại container đang dùng nếu nó đã đáp ứng preflight. Dockerfile mới chưa được build/test trên máy hiện tại.

## 3 Chuẩn bị và pilot

```bash
python run_all.py --stage prepare --source-root /workspace/rag-inputs
python run_all.py --stage pilot --device cuda --source-root /workspace/rag-inputs
```

Pilot chạy 3 câu dev mỗi cấu hình; không phải benchmark đủ 100 câu. Xem `runs/pilot/*/predictions.jsonl` để phát hiện output rỗng, lặp prompt, nguồn không hợp lệ, OOM hoặc sinh không đúng tiếng Việt. `--stage pilot` vẫn xây dense index toàn corpus để kiểm tra đúng pipeline thật; bước này có thể lâu hơn phần sinh.

Lần tải đầu cần Internet đến Hugging Face và PyPI. Không cần API inference trả phí. Model revisions được khóa trước khi tải; nếu lock đã có sẽ giữ nguyên.

## 4 Chạy đầy đủ

Trong bash, dùng pipefail để lỗi pipeline không bị `tee` che mất:

```bash
set -o pipefail
python -u run_all.py --stage full --device cuda --source-root /workspace/rag-inputs 2>&1 | tee p100_full.log
```

Nếu muốn thử legal chunking, quyết định trước khi xem test:

```bash
python -u run_all.py --stage full --device cuda --include-legal --source-root /workspace/rag-inputs 2>&1 | tee p100_full.log
```

Mặc định: xây index baseline một lần, chạy B0/B1/R1/R2 trên 107 câu dev, chọn theo dev, chạy B0/B1 và cấu hình được chọn trên 100 câu test. Nếu cấu hình được chọn trùng baseline thì không chạy lặp và không tuyên bố có cải thiện. Không chạy fine-tuning.

## 5 OOM, ngắt phiên và phục hồi

- OOM khi embedding: chạy lại với `--batch-size 16` hoặc `8`; checkpoint giữ các vector đã hoàn tất. Batch size không thay nội dung embedding, nhưng có thể gây sai khác số học nhỏ theo backend; phải giữ log.
- OOM generation: giữ run lỗi để audit. Tạo cấu hình mới với input budget nhỏ hơn và tên khác; chạy lại các baseline/candidate với cùng ngân sách. Không âm thầm đổi chỉ một hệ thống trong phép so sánh.
- Ngắt phiên: chạy lại đúng lệnh. Các run hoàn tất được nhận diện; câu đã ghi được giữ.
- JSONL bị ghi dở do kill: dùng `python scripts/repair_jsonl.py path/to/predictions.jsonl`. Công cụ chỉ loại dòng cuối không hợp lệ, lưu backup và ghi nhật ký; không loại dòng lỗi ở giữa hoặc dòng có trường `error` hợp lệ.
- Mẫu có trường `error` vẫn là mẫu đánh giá và chấm 0. Xem lỗi trước khi diễn giải điểm thấp.

Nếu phát hiện lỗi logic sau khi đã xem test, báo rõ và tạo split mới cho lần xác nhận độc lập. Không tune tiếp trên test đã xem.

## 6 Gói gửi lại

Sau full, bộ chạy tự tạo ZIP. Có thể tạo lại bằng:

```bash
python run.py report
python run.py package --kind results
```

Gửi `dist/rag_assignment_results.zip` và `p100_full.log`. ZIP chứa audit, manifest, selection, predictions/scores/summary, so sánh và báo cáo; không chứa model weights hay corpus SQLite đầy đủ.

Nếu dừng ở pilot, gửi ZIP pilot và log trước để kiểm tra lỗi rồi mới chạy full. Không cần gửi token truy cập hoặc thông tin đăng nhập server.

## Nguồn tương thích

- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- PyTorch Pascal/CUDA packaging: https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128
- NVIDIA architecture matrix: https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
