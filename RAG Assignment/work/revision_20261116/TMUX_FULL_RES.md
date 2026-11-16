# Chạy baseline trên toàn bộ RES trong tmux

## Phạm vi đã chốt

Đánh giá chính dùng toàn bộ 2.745 cặp của RES, theo thứ tự dòng Excel trong full_res_selection.json. Các phương pháp đề xuất phải dùng đúng cùng ID và bộ chấm. Tập này bao gồm các câu đã được xem ở bước phát triển; không gọi đây là một holdout độc lập.

Benchmark 30 câu người học gửi: 0 lỗi, 0 đáp án rỗng, 0 chuỗi chứa prompt, 0 vi phạm cấu hình sinh, tối đa 256 token sinh; vòng đánh giá khoảng 5 phút 33 giây và toàn lần chạy 355,20557038113475 giây. Tốc độ hiển thị 11,12 giây/câu cho dự báo khoảng 8 giờ 30 phút với 2.745 câu nếu điều kiện tương tự. Đây là dự báo từ 30 câu, chưa phải thời gian thực đo của lần chạy toàn bộ. Thời gian xây FAISS không được cộng lại vì dùng chỉ mục pilot 01 đã lưu.

## Đồng bộ source trên Windows

```powershell
cd "D:\master_hcmut\HK261\Large_Language_Model\RAG Assignment"
git add -- "work/revision_20261116"
git commit -m "Add full RES evaluation and tmux launcher"
git push origin main
```

## Tạo tmux trên host server

Tại prompt `[baond12@cmr-node04-dc9c15u21 ~]$`, chạy:

```bash
tmux new -s rag-baseline-full
```

tmux chạy trên host; Python vẫn chạy bên trong container baond_llm. Trong cửa sổ tmux mới, chạy:

```bash
docker exec -i \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e http_proxy=http://proxy.hcm.fpt.vn:80 \
  -e https_proxy=http://proxy.hcm.fpt.vn:80 \
  -w "/workspace/Large_Language_Models_Course_HK261" \
  baond_llm bash -c \
  'git pull --ff-only origin main &&
   cd "RAG Assignment/work" &&
   bash "revision_20261116/Source code/run_full_res.sh" instructor_baseline_full_2745_01'
```

Script chọn `--evaluation-set full-res`, không đặt limit, nên đọc đủ 2.745 ID trong manifest. Kiểm tra đầu ra ban đầu phải có `"questions": 2745` và thông báo dùng lại chỉ mục đã kiểm tra. Không lấy lại chỉ mục custom từ trước. Tên lần chạy và log phải mới, script sẽ từ chối ghi đè.

## Tách phiên và xem lại

Nhấn Ctrl+B, thả tay rồi nhấn D để detach. Chương trình tiếp tục trong tmux khi đóng SSH. Sau khi đăng nhập lại:

```bash
tmux attach -t rag-baseline-full
```

Hướng dẫn thao tác tmux tham khảo tài liệu chính thức: https://github.com/tmux/tmux/wiki/Getting-Started

## Log và kết quả

Trong thư mục work của container:

- Log liên tục: `artifacts/notebook_execution_20261116/instructor_baseline_full_2745_01.console.log`.
- Notebook đã thực thi và log notebook: `artifacts/notebook_execution_20261116/instructor_baseline_full_2745_01/`.
- Câu hỏi, câu trả lời, điểm, cấu hình và summary: `runs/notebook_baseline_20261116/instructor_baseline_full_2745_01/`.

Sau khi hoàn tất, yêu cầu summary có questions=2745, expected_selection_count=2745, evaluation_set=full-res, limited_run=false, errors=0, returned_prompt_prefix_count=0 và generation_contract_violations=0. Kiểm tra thêm số bản ghi, danh sách ID, hash và tính lại điểm từ các tệp thực tế trước khi đưa vào report. Chạy đủ không đồng nghĩa mọi câu trả lời đều đúng.

tmux giữ tiến trình qua việc tách/kết nối lại SSH; script ghi stdout/stderr liên tục và giữ exit code lỗi của Python. Nó không tự tạo ZIP hay nộp bài. Bộ so sánh phương pháp đề xuất và nội dung report đầy đủ sẽ được kiểm tra sau khi có kết quả, rồi đưa người học review trước khi đóng gói.
