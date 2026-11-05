# Kịch bản demo

Demo cần cho thấy câu hỏi → các đoạn truy hồi → câu trả lời → nguồn và giới hạn. Chọn cấu hình đã thắng trên dev, không mặc định R2 chắc chắn tốt hơn. Thay đường dẫn config trong các lệnh sau theo `artifacts/selection.json`.

1. Giới thiệu dữ liệu, giới hạn LLM và sơ đồ chunk → BM25/dense → RRF → context → Qwen.
2. Chạy câu có bằng chứng thực tế. Sau retrieval benchmark, chọn một câu từ diagnostic_dev và mở tệp nguồn để xác minh nội dung trước buổi demo. Không lấy nhãn tham chiếu làm đầu vào truy hồi.
3. Chạy câu nhiều yêu cầu, xem đủ ý, số liệu, điều kiện và nguồn.
4. Chạy câu thiếu thông tin hoặc ngoài corpus, kiểm tra hệ thống có nói rõ thiếu bằng chứng không.
5. Trình bày bảng test đủ 100 câu, cả cải thiện và suy giảm; mở một ca lỗi.

Ví dụ lệnh, chưa mặc nhiên là ví dụ được trả lời đúng:

```bash
python run.py demo --config configs/r2_grounded.json --device cuda --question "Văn bản được truy hồi quy định điều kiện và đối tượng áp dụng chính sách như thế nào?"
python run.py demo --config configs/r2_grounded.json --device cuda --question "Theo chính sách nội bộ Công ty Sao Hỏa năm 2099, tôi được hưởng bao nhiêu ngày nghỉ phép?"
```

Tệp `artifacts/last_demo.json` lưu nguồn, token budget và output. Nhãn [S#] hợp lệ về cú pháp chưa đủ chứng minh câu trả lời đúng; người trình bày phải đối chiếu nội dung nguồn. Trước khi nộp, thay câu hỏi minh họa chung bằng ví dụ cụ thể đã kiểm chứng từ corpus và giữ cả ví dụ thất bại.
