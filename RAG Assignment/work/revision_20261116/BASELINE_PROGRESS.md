# Tiến độ baseline theo log thực tế người học cung cấp

Nguồn: log từ server của lần `instructor_pilot_1q_01`, nhận trong phiên làm việc hiện tại. Đây là ghi nhận một lần chạy thất bại ở bước nạp LLM, không phải kết quả đánh giá đã hoàn thành.

- GPU vật lý số 3, Tesla P100-PCIE-16GB, Python 3.10.13, Transformers 4.51.3.
- Nạp 16.440 văn bản, tạo 176.506 đoạn.
- Vòng embedding hoàn tất 173/173, thời gian hiển thị 54 phút 02 giây.
- Lưu và nạp lại FAISS thành công; cell 3 và 4 đều báo `ok`.
- IndexFlatL2, không chuẩn hóa L2, 176.506 vector, chiều 768. Đây là cấu hình đã quan sát khi chạy nhánh notebook; không gọi nó là cosine.
- SHA256 index.faiss: `8a20b51bb72a3871c5047c937d790e2cfcafc7ef94e13fc9337903ce17779428`.
- SHA256 index.pkl: `c86ebefaa914c776efff4cbc1a5fb763d60dc2835a9204dad4ed04cde69cb11d`.
- Cell 5 lỗi tại HuggingFacePipeline.from_model_id vì revision bị truyền trùng. Chưa có câu trả lời hay điểm số cho pilot này.

Sửa ở bản Docker: khai báo framework PyTorch trong pipeline_kwargs; không thay prompt, mô hình, cách sinh, metric hoặc phần tạo FAISS. Lưu nguyên lần thất bại và chạy lại với tên mới, dùng chỉ mục cũ qua cơ chế kiểm tra hash. Kiểm thử tương thích CPU là bằng chứng cho đường gọi pipeline; chưa chứng minh Qwen chạy thành công trên P100.

Log đầy đủ đã được giữ riêng tại `artifacts/notebook_execution_20261116/received_instructor_pilot_1q_01.txt` trong workspace. Khi hoàn thiện bộ bài nộp cần lấy thêm executed.ipynb, environment.json, notebook.log và index_info.json trực tiếp từ server để đối chiếu với log người học gửi.
