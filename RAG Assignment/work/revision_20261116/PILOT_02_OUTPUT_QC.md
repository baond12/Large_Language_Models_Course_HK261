# Kiểm tra đầu ra pilot 02

Nguồn: log người học gửi từ `instructor_pilot_1q_02`, giữ nguyên tại `artifacts/notebook_execution_20261116/received_instructor_pilot_1q_02.txt`. Đây là một câu thử, chưa phải đánh giá tối thiểu 100 cặp.

## Những gì đã quan sát

- Notebook hoàn tất các cell 0–5 trên GPU vật lý số 3; lỗi duplicate revision đã được vượt qua.
- Lần chạy dùng lại chỉ mục FAISS của pilot 01, không xây lại embedding.
- Có 1 bản ghi, errors=0, empty_answers=0, returned_prompt_prefix_count=1.
- Chuỗi in ra gồm toàn bộ hướng dẫn, năm đoạn ngữ cảnh, câu hỏi và phần mô hình sinh thêm. Hàm metric đang chấm cả chuỗi này.
- Câu hỏi có ID q_a3de7687e2845f64. Các đoạn được in chủ yếu về tuyển dụng công chức/viên chức, trong khi câu hỏi hỏi điều kiện bổ nhiệm kế toán trưởng tổ chức tín dụng. Phần sinh lặp các mẫu câu về điều kiện theo số điều; chưa có căn cứ để công nhận trả lời đúng.
- Log báo elapsed_seconds=131.64065700024366; predictions_sha256 do server báo là 8595b3753519f74bf6a4368318ba9502a2d652b57730fadd952da569b967d159. Cần lấy tệp từ server để tự tính lại hash và điểm khi hoàn thiện bằng chứng.

## Quyết định về điểm số

Giữ nguyên các điểm đã ghi trong pilot 02 như bằng chứng của đường chạy cũ, không đưa chúng vào bảng điểm câu trả lời của report. Một lần chạy không lỗi runtime không chứng minh phép chấm đúng đối tượng. Không gọi sửa lỗi định dạng đầu ra là cải tiến retrieval hoặc generation.

## Sửa có căn cứ

Mã nguồn chính thức của langchain-huggingface 0.3.1, đọc từ wheel PyPI và kiểm tra SHA256, cho thấy method `_generate` chỉ truyền kwargs['pipeline_kwargs'] xuống pipeline. Vì vậy tham số max_new_tokens=256 ở mức ngoài trong lời gọi cũ chưa được chuyển tới backend. Chưa có bản ghi số token backend của pilot 02 để kết luận chính xác nó sinh bao nhiêu token.

Đổi riêng lời gọi trong rag_query sang `llm.generate([prompt], pipeline_kwargs={"max_new_tokens": 256, "return_full_text": False})`. Không sửa prompt, retriever, model, chỉ mục hay công thức metric. Thêm observer không thay đổi output để lưu actual input_tokens, generated_tokens và effective_max_new_tokens ở backend. Bản mới phải thử lại trên P100; chỉ khi errors=0, returned_prompt_prefix_count=0 và generation_contract_violations=0 mới chuyển sang 100 câu.

Giữ tên lần chạy mới, dùng lại index đã xác minh của pilot 01. Nêu rõ thay đổi API này trong source và thiết lập thực nghiệm cuối; áp dụng cùng cách lấy câu trả lời và cùng evaluator cho các phương pháp đề xuất.
