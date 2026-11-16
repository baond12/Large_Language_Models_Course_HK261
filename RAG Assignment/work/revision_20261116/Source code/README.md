# Chạy notebook baseline của giảng viên trong Docker

## Trạng thái và phạm vi

Đây là bản điều chỉnh từ notebook giảng viên, chưa có lần thực thi GPU được kiểm chứng trong đợt ngày 16/11/2026. Không thay đường chạy bằng pipeline RAG tự viết trước đây. Kết quả baseline mới chỉ được ghi vào report sau khi notebook thực thi đủ ít nhất 100 cặp và đã kiểm tra đầu ra.

Notebook gốc ở `../../../Baseline.ipynb` tính từ thư mục này được giữ nguyên. Công cụ tạo bản Docker đối chiếu SHA256 của bản gốc trước khi làm việc. Hai file `.diff` liệt kê thay đổi từng cell để người đọc kiểm tra.

Hai bản được tạo:

- `Baseline_Docker_notebook.ipynb`: giữ cách xây và tìm kiếm FAISS của notebook, chưa đặt thêm distance/normalization. Loại khoảng cách thực tế sẽ được ghi từ index sau khi chạy.
- `Baseline_Docker_cosine.ipynb`: vẫn dùng các cell và hàm notebook; thêm chuẩn hóa L2 và tìm bằng inner product để xếp hạng cosine theo mô tả đề. Phải xây index riêng và không trộn kết quả với nhánh trên.

Nếu hai nhánh tương đương trong môi trường đã xác nhận, sẽ trình bày rõ và tránh nhận chúng là hai cải tiến khác nhau. Hiện chưa có bằng chứng chạy GPU cho cả hai.

## Những thay đổi đã kê khai

1. Thay Google Drive, `/content` và tải Google Sheets bằng các bản Dataset.zip, RES.xlsx được cấp. Bản XLSX và ZIP được ghi hash; không khẳng định XLSX bằng một phiên bản Google Sheets đang trực tuyến nếu chưa đối chiếu.
2. Dùng đúng 100 ID đã khóa trong hợp đồng thực nghiệm thay cho `head(30)`. Xuất riêng CSV có các câu hỏi, đáp án và dòng Excel để kiểm tra. Đề cho phép chọn tập con ít nhất 100 cặp, không bắt buộc 100 dòng đầu.
3. Bỏ các cell tự động pip install/upgrade và import Google API không được sử dụng. Công cụ kiểm tra thư viện trước; chỉ cài phần thiếu sau khi xem phiên bản hiện có. Không tạo môi trường Python riêng trong Docker.
4. Khóa revision của Vietnamese SBERT và Qwen1.5-1.8B theo models.lock.json; đặt seed 261. Đây là thiết lập cho lần chạy mới, không phải bằng chứng về seed hoặc revision của một lần chạy cũ của giảng viên.
5. Giữ thứ tự tệp thu được ở lần đọc đầu tiên và lưu danh sách đó để chạy lại, không tự sắp xếp lại trong loop. Khi tái lập ở thư mục sạch, cung cấp danh sách từ lần thực nghiệm bằng biến BASELINE_DOCUMENT_ORDER.
6. Thêm lưu cấu hình, prompt, đoạn truy hồi, raw response, lỗi, điểm từng câu, notebook đã thực thi và phiên bản thư viện. Các lệnh quan sát không sửa prompt hoặc response.
7. Nhánh cosine có thay đổi khoảng cách/chuẩn hóa được nêu rõ. Nhánh notebook không có thay đổi đó.

Các hàm chuẩn hóa, rag_query và metric được lấy từ nguồn notebook. Kiểm tra AST xác nhận tám hàm này giữ nguyên sau khi loại lời gọi ghi log được bổ sung trong rag_query. Vẫn giữ chunk 2000/50, top 5, prompt gốc, max_new_tokens=256, cách gọi HuggingFacePipeline và cách trả về chuỗi của notebook. Không thêm ngân sách đầu vào 4096, cắt completion, đổi FP16, đổi decoding hoặc tối ưu lại vòng embedding trong lần chuẩn bị này. Điều này cũng có nghĩa là chưa thể cam kết phần gọi mô hình sẽ chạy trên môi trường hiện có; nếu có lỗi phải lưu và sửa theo bằng chứng.

## Bước hiện tại trên server — chỉ kiểm tra môi trường

Sau khi đưa các file mới lên repository và cập nhật bản clone trong Docker, chạy từ shell Linux trên server:

```bash
docker exec -it \
  -e CUDA_VISIBLE_DEVICES=3 \
  -w "/workspace/Large_Language_Models_Course_HK261/RAG Assignment/work" \
  baond_llm python3 "revision_20261116/Source code/run_instructor_notebook.py" --check
```

Lệnh này chưa xây index, tải mô hình hoặc đánh giá 100 câu. Nó báo thư viện thiếu, lỗi import thực tế, dữ liệu đầu vào và CUDA. Gửi đầy đủ kết quả để quyết định chính xác thư viện nào cần cài; không tự nâng cấp torch/transformers/langchain cả bộ.

## Thực thi sau khi môi trường đã qua kiểm tra

Theo yêu cầu chạy thử trước: thêm `--limit 1` để thực thi một câu bằng notebook. Lần đầu vẫn phải xây FAISS trên toàn bộ corpus để giữ đúng truy hồi baseline. Pilot được đánh dấu limited_run và không đáp ứng yêu cầu đánh giá tối thiểu 100 cặp.

Sau khi kiểm tra output và log của một câu, chạy cùng nhánh với tên lần chạy mới, bỏ `--limit`, thêm `--reuse-index-from <tên-lần-pilot>`. Công cụ đối chiếu dữ liệu, revision, logic tạo index, thư viện truy hồi và hash hai tệp FAISS trước khi dùng lại. Chỉ mục là chỉ mục do chính notebook vừa tạo, không phải chỉ mục từ pipeline tự viết trước đây. Luồng sinh câu trả lời và các hàm metric vẫn giữ nguyên. Runner hiển thị stdout và tiến độ khi notebook đang chạy.

Các lệnh dưới là hướng dẫn chuẩn bị cho bước sau, chưa được xác nhận thực chạy GPU. Dùng tên lần chạy mới để bảo toàn log cũ.

```bash
docker exec -it \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e http_proxy=http://proxy.hcm.fpt.vn:80 \
  -e https_proxy=http://proxy.hcm.fpt.vn:80 \
  -w "/workspace/Large_Language_Models_Course_HK261/RAG Assignment/work" \
  baond_llm python3 "revision_20261116/Source code/run_instructor_notebook.py" \
    --execute --mode notebook --run-name instructor_notebook_attempt01
```

Sau khi đọc log, xác nhận index, cấu hình và raw response của nhánh notebook, mới chạy nhánh cosine bằng cùng lệnh với `--mode cosine --run-name instructor_cosine_attempt01`. Nhánh này xây index riêng, không dùng chỉ mục custom trước đây. Hai lần chạy có thể tốn thời gian vì giữ vòng embedding của notebook.

Notebook được chạy thật qua Jupyter kernel bằng nbclient. Mỗi cell hoàn tất được lưu vào notebook thực thi; lỗi tạo failure_traceback.txt và notebook.log, rồi chương trình dừng với exit code lỗi. Các lỗi từng câu mà notebook bắt lại vẫn được ghi traceback và giữ trong đủ N bản ghi, không được nhận là đường chạy đã nghiệm thu thành công. Không sửa lỗi suy đoán trước khi có log.

## Đầu ra để đối chiếu

Đầu ra thực nghiệm ở `runs/notebook_baseline_20261116/<run-name>`:

- evaluation_pairs.json và evaluation_res.csv: đúng tập câu hỏi, đáp án và ID cố định.
- environment.json, document_order.json, model_info.json, index_info.json: đầu vào và cấu hình thực tế.
- predictions.jsonl: câu hỏi, đáp án, raw response đúng chuỗi notebook trả về, context và lỗi.
- rag_baseline_metrics.csv: điểm các hàm notebook, không thay định nghĩa.
- scores_pdf.json và summary.json: xuất thêm biến thể set trong PDF và tổng hợp; chưa phải xác nhận chạy lại độc lập.

Bằng chứng notebook và môi trường ở `artifacts/notebook_execution_20261116/<run-name>`: executed.ipynb, notebook.log, execution_status.json, pip-freeze.txt và failure_traceback.txt nếu có lỗi. Tiến độ của một cell dài được cập nhật trong kernel; bản notebook ngoài chỉ lưu sau khi cell hoàn tất.

## Chạy lại và nghiệm thu

### Lỗi truyền trùng revision đã quan sát trên server

Lần `instructor_pilot_1q_01` đã xây và lưu xong FAISS gồm 176.506 vector trên 16.440 văn bản, rồi lỗi tại cell nạp LLM: `infer_framework_load_model() got multiple values for keyword argument 'revision'`. Tham số revision được thêm trong bản Docker để khóa mô hình; không quy lỗi này cho notebook gốc chưa có tham số đó.

Sửa tương thích: thêm `pipeline_kwargs={"framework": "pt"}` vào lời gọi `HuggingFacePipeline.from_model_id`. LangChain đã nạp model PyTorch; chỉ định rõ framework giúp Transformers không đi qua nhánh tự suy luận framework đang nhận revision trùng. Giữ model ID, model_kwargs (temperature, max_length, revision), device, prompt, max_new_tokens và các hàm metric. Chưa đổi dtype hoặc thuật toán truy hồi.

Lỗi và cách sửa đã được tái hiện bằng Transformers 4.51.3 với một mô hình nhỏ khởi tạo ngẫu nhiên trên CPU, không tải mô hình và không coi đây là kết quả RAG. Xác nhận Qwen/P100 vẫn cần lần thử lại thực tế. Dùng tên mới `instructor_pilot_1q_02` và `--reuse-index-from instructor_pilot_1q_01 --limit 1` để giữ và kiểm tra chỉ mục đã lưu; không chạy lại embedding.

Hướng dẫn tái lập cuối phải dùng các phiên bản đã thực chạy thành công, không chỉ danh sách thư viện cài được. Hiện chưa phát hành file requirements mang nhãn “đã kiểm chứng” cho notebook này.

Khi chạy lại ở thư mục sạch, dùng đúng model revisions, dữ liệu, thứ tự câu và danh sách document_order.json từ lần đã báo cáo. Truyền danh sách tệp bằng `-e BASELINE_DOCUMENT_ORDER="<đường dẫn trong container tới document_order.json đã báo cáo>"` trong docker exec. Không truyền đường dẫn Windows cho tiến trình Linux.

Phải so sánh từng ID, câu trả lời và điểm số với lần được báo cáo. Bất kỳ khác biệt chưa giải thích được đều cần ghi nhận và xử lý trước khi nhận đạt khả năng tái lập. Chỉ chấm lại predictions.jsonl hoặc mở executed.ipynb không thay thế phép chạy lại này.

Source cho các phương pháp đề xuất sẽ dùng cùng evaluation_pairs.json và cùng evaluator. Chưa có kết quả so sánh mới, vì vậy report đang để trống các ô định lượng. Không đóng gói ZIP cuối trước khi người học review và chốt.
