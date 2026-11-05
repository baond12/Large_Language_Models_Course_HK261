# Trạng thái bàn giao code

## Đã thực hiện local

- Đọc đề và baseline; khảo sát ghi chú tuần 1–7, exercise/solution, outline slides và cấu trúc lab liên quan.
- Tạo kế hoạch, ma trận yêu cầu–bằng chứng và audit sửa lỗi baseline.
- Đọc 16.440 văn bản và 2.745 cặp hỏi–đáp, lưu fingerprint, kiểm tra nhãn thiếu và coverage số hiệu.
- Tạo split chính dev 107/test 100; diagnostic_dev 29/diagnostic_test 22 độc lập theo số hiệu parse được.
- Tái tạo 176.506 chunk baseline 2000/50, khớp số lượng lưu trong notebook gốc.
- Xây BM25 toàn corpus và đo trên diagnostic_dev. Đây là retrieval diagnostic, không phải metric generation của đề.
- Viết B0/B1/R1/R2 và R3 tùy chọn, evaluator, checkpoint/resume, bộ chạy CLI, notebook, Dockerfile và hướng dẫn server.
- Kiểm thử metric so với code gốc, FAISS, mô hình tiny cục bộ, đọc ZIP/XLSX, split, scoring/comparison và giữ lỗi trong mẫu số. Log ở `artifacts/tests.log`, trạng thái máy đọc được ở `artifacts/validation.json`.

## Đã triển khai nhưng cần chạy trên P100

- Tải/khóa hai mô hình thật và encode dense toàn corpus.
- Pilot 3 câu mỗi cấu hình, sau đó dev đầy đủ và chọn cấu hình.
- Chạy cùng 100 câu test cho baseline và hệ thống cuối, chấm đủ 5 metric.
- So sánh cải thiện và độ trễ/VRAM trên P100; phân tích nội dung các ca tăng/giảm.
- Chạy và duyệt demo, hoàn thiện báo cáo với kết quả thật.

## Giới hạn cần giữ trong báo cáo

Chỉ 51 câu có số hiệu tham chiếu khớp tên tệp corpus sau chuẩn hóa; 779 câu không có nhãn số hiệu parse được. Đây là phép đối chiếu tên tệp, không phải kết luận đáp án vắng toàn corpus. Không thay tập test chính bằng tập covered để tăng điểm mà không công bố. Chưa có cơ sở tuyên bố cải thiện metric generation hoặc bài đã đủ điều kiện nộp.

Theo phân công của người học: người học chạy trong Docker P100 rồi gửi `dist/rag_assignment_results.zip` cùng log; bước tiếp theo là kiểm tra kết quả thật và hoàn thiện phân tích.

## Cập nhật sau pilot trên server

Đã nhận log: dense encode đủ 176.506 vector; retrieval diagnostic_dev 29 câu có Hit@5 B0=0.275862, B1=0.310345, BM25=0.793103, hybrid=0.482759. Pilot 3 câu của B0/BM25/R2 trên GPU 3 hết lỗi OOM nhưng vẫn lặp nội dung và chạm 256 token. Đây là kết quả người học gửi, tệp gốc vẫn trên server; chưa có benchmark generation >=100 câu.

Đã bổ sung nhánh Instruct với chat template, bảo toàn baseline và thiết kế ablation tại `docs/instruct_experiment.md`. Chưa chạy mô hình Instruct thật trên P100; cần cập nhật code server và pilot trước. Người học dùng trực tiếp Docker, không tạo venv; chỉ cài dependency còn thiếu.
