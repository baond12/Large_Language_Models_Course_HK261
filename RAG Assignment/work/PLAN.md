# Kế hoạch thực hiện RAG Assignment

## Mục tiêu và môi trường

Nguồn quyết định: `../RAG_Assignment.pdf`, phiên bản 1.0 ngày 09/11/2025. Ngày triển khai theo phiên làm việc: 05/11/2026. Ngày trên đề là ngày phiên bản, không phải deadline. Deadline thực tế chưa được xác nhận.

Người học xác nhận có GPU **P100 16 GB trên máy/server riêng**; máy Windows hiện tại có khoảng 32 GB RAM, Intel Arc Pro Graphics và Python 3.13. Không giả định đã có quyền kết nối server. Phần chuẩn bị dữ liệu và kiểm thử chạy local; phép đo mô hình đầy đủ cần môi trường P100. Không dùng dịch vụ trả phí.

## Trình tự và điều kiện nghiệm thu

| Bước | Việc làm | Bằng chứng | Điều kiện hoàn tất |
|---|---|---|---|
| 1 | Audit đề, notebook, RES và corpus | `docs/requirements.md`, `docs/baseline_audit.md`, `artifacts/data_audit.json` | Mỗi yêu cầu có nguồn, cách kiểm tra; đếm dữ liệu thật |
| 2 | Kiểm tra chất lượng dữ liệu; đóng băng split theo nhóm văn bản | `data/questions.jsonl`, `data/labels.jsonl`, `data/splits.json` | Không trùng ID hoặc nhóm văn bản giữa dev/test; test >=100; có fingerprint |
| 3 | Kiểm thử metric và các thành phần độc lập mô hình | `tests/`, `artifacts/tests.log` | Ví dụ metric kiểm tra tay, không đọc nhãn khi inference, context vừa ngân sách |
| 4 | Tạo baseline và các ablation | `configs/`, `src/rag_assignment/` | Giữ 2000/50, Vietnamese SBERT, top-5, Qwen1.5-1.8B; ghi riêng L2/cosine |
| 5 | Chạy retrieval trên dev, chọn phương án | `artifacts/retrieval_dev/`, `artifacts/selection.json` | Chọn bằng dev, không xem nhãn test để chỉnh cấu hình; báo coverage nhãn |
| 6 | Chạy generation dev và test trên P100 | `runs/`, `artifacts/comparison.json` | Cùng ID test >=100, cùng evaluator, đủ 5 metric; giữ cả mẫu lỗi |
| 7 | Phân tích lỗi, bootstrap ghép cặp, đo tài nguyên | `artifacts/comparison.json`, `artifacts/error_analysis.md` | Nêu tăng/giảm và giới hạn, không chỉ chọn ví dụ tốt |
| 8 | Demo, báo cáo, README và đóng gói | `RAG_Assignment.ipynb`, `REPORT.md`, `DEMO.md`, `DEFENSE.md` | Chạy từ đầu hoặc quy trình CLI tương đương; phân biệt đã chạy/chưa chạy |

## Thứ tự thử nghiệm

1. **B0 baseline code**: chunk 2000/50, embedding không normalize, FAISS squared-L2, top-5, prompt gốc, Qwen1.5-1.8B. Sửa lỗi vận hành và lấy completion riêng; ghi rõ giới hạn context chung.
2. **B1 baseline mô tả đề**: giữ B0, đổi sang cosine bằng normalize + inner product. Tránh quy toàn bộ cải thiện cho một khác biệt giữa code và đề.
3. **R1 hybrid cùng chunk/prompt/LLM**: BM25 + dense cosine với reciprocal rank fusion (RRF). Giả thuyết: thuật ngữ và số hiệu pháp luật cần lexical matching; embedding bổ sung ngữ nghĩa. Đo Hit@5/Recall@5/MRR trên nhãn văn bản yếu và 5 metric câu trả lời.
4. **R2 context/prompt**: giữ retriever R1, đóng gói theo ngân sách token, nguồn rõ, yêu cầu trả lời đủ ý và không bịa nguồn. So với R1 để tách tác dụng prompt.
5. **R3 legal chunking**: chunk theo Điều, giữ tiêu đề và nguồn, giới hạn kích thước. Chỉ chọn nếu cải thiện dev tương xứng chi phí xây index bổ sung.

Phương án tối thiểu: B0/B1 và một pipeline cải tiến có đo generation >=100 cặp. Không bắt buộc fine-tuning, đổi LLM hoặc reranking neural. Chỉ mở rộng sau khi các bước chính có bằng chứng.

## Quy tắc quyết định

- Chọn retrieval bằng MRR@5 trên dev có nhãn khớp corpus; nếu coverage thấp, coi là chẩn đoán và ưu tiên metric generation dev.
- Chọn hệ thống cuối bằng ROUGE-L dev (metric chính khai báo trước), dùng các metric khác và độ trễ làm thông tin phụ. Không chọn theo test.
- Nếu không có cải thiện: giữ kết quả thật, phân tích lỗi, sửa trên dev và chạy lại vòng test chỉ khi ghi rõ test đã được xem và phải lập tập holdout mới để có xác nhận độc lập.
- Không tuyên bố assignment hoàn tất khi chưa có phép đo baseline và hệ thống cuối trên cùng ít nhất 100 cặp.
