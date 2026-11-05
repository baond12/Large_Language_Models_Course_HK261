# Ma trận yêu cầu và bằng chứng

Đề là nguồn bắt buộc. Các đề xuất tái lập và kiểm thử bên dưới là biện pháp kỹ thuật bổ sung, không tự gán thêm yêu cầu cho thầy.

| ID | Loại | Yêu cầu | Nguồn | Điểm | Thực hiện và bằng chứng | Kiểm tra |
|---|---|---|---|---|---|---|
| REQ01 | Bắt buộc | RAG hỏi đáp luật; tập trung retrieval và context | PDF §1–2, tr.1–2 | — | Code retriever, context, generator; mô tả trong báo cáo | Inference chỉ nhận câu hỏi và corpus |
| REQ02 | Bắt buộc | LLM ít hơn 4 tỷ tham số | PDF §3, tr.3–4 | — | Qwen/Qwen1.5-1.8B; lưu số tham số thực tế | Runtime từ chối model >=4,000,000,000 tham số |
| REQ03 | Bắt buộc | Đánh giá ít nhất 100 cặp; báo số lượng | PDF §3.1, tr.3 | — | Split cố định; predictions và metrics từng câu | So sánh trên cùng >=100 ID, không bỏ lỗi |
| REQ04 | Bắt buộc | Cải thiện trên ít nhất một metric được nêu | PDF §4, tr.5 | 3 | Bảng baseline/cải tiến: cosine, token overlap/Jaccard, BLEU, ROUGE-L | Tính từ predictions đã lưu, cùng evaluator và tập test |
| REQ05 | Bắt buộc | Demo rõ, liên quan, đúng và dễ chạy | PDF §4, tr.5 | 2 | Notebook/CLI + `DEMO.md` | Chạy ví dụ thường, khó, thiếu nguồn; kiểm tra nội dung |
| REQ06 | Bắt buộc | Khám phá và hiểu dữ liệu | PDF §4, tr.6 | 1 | Audit số lượng, độ dài, nhóm văn bản, coverage, chất lượng | Dữ liệu thực tế và định nghĩa thống kê được lưu |
| REQ07 | Bắt buộc | Giải thích cải tiến và có bằng chứng định lượng | PDF §4, tr.6 | 3 | Thiết kế ablation, kết quả và phân tích lỗi | So sánh kiểm soát thành phần thay đổi |
| REQ08 | Bắt buộc | Báo cáo có cấu trúc, rõ và tái lập được | PDF §4, tr.6 | 1 | Báo cáo, bảng/biểu đồ nếu hữu ích, README | Đối chiếu từ lệnh chạy → kết quả → nhận định |
| ENG01 | Bổ sung | Không rò rỉ nhãn | Thiết kế thực nghiệm | — | Câu hỏi và nhãn ở tệp riêng; nhóm văn bản không giao nhau giữa dev/test | Test cấu trúc inference; assert split |
| ENG02 | Bổ sung | Tái lập/công bằng | Thiết kế thực nghiệm | — | Seed, hash dữ liệu/cấu hình, phiên bản model và dependency, completion-only | Kiểm tra manifest, resume và ID |
| ENG03 | Bổ sung | Hiệu quả tài nguyên | Mục tiêu người học | — | Thời gian build/query/generation và peak memory | Nêu phần cứng, warm-up và phạm vi phép đo |
| ENG04 | Bổ sung | Hiểu và bảo vệ bài | Mục tiêu người học, ghi chú tuần 7 | — | `DEFENSE.md` và giải thích code | Trả lời vì sao kỹ thuật cải thiện hoặc không cải thiện |

Ghi chú môn học: đã khảo sát ghi chú tuần 1–7, outline slides trong `slides/`, một số exercise/solution LM/MLM/self-attention và cấu trúc notebook lab RNN, Seq2seq, Transformer QA/NMT, PhoBERT. Đây là ngữ cảnh sư phạm, không phải rubric bổ sung. Các tệp lab đã có chỉnh sửa của sinh viên nên không xem mọi đoạn code là mẫu chuẩn của thầy.
