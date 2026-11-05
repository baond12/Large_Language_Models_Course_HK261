# Thử nghiệm mô hình Instruct sau pilot

## Bằng chứng và giả thuyết

Ngày 05/11/2026, người học chạy trên Docker Python 3.10, GPU P100 số 3. Trên cùng 3 câu dev, B0, BM25 với prompt baseline và R2 đều không có exception nhưng lặp nội dung, cả 3 câu chạm 256 token. Câu trả lời BM25 thứ hai mâu thuẫn với reference được cấp và tiếp tục thành hội thoại nhiều ngôn ngữ không liên quan. R2 không dùng nhãn [S#]. Đây là quan sát pilot, không suy rộng thành kết quả toàn dev/test.

Điểm pilot người học cung cấp:

| Hệ thống | N | Cosine | ROUGE-L | Chạm giới hạn output |
|---|---:|---:|---:|---:|
| B0 code, Qwen1.5-1.8B | 3 | 0.677618 | 0.244904 | 3 |
| BM25, Qwen1.5-1.8B | 3 | 0.668081 | 0.190045 | 3 |
| R2, Qwen1.5-1.8B | 3 | Chưa chấm | Chưa chấm | 3 |

Các số trên được chép từ log người học, chưa có bản sao JSONL/summary tương ứng trên máy Windows. Các tệp gốc vẫn ở server. Không đưa số pilot này vào bảng test >=100 câu.

Qwen1.5-1.8B là bản base. Model card khuyến nghị dùng hậu huấn luyện cho sinh văn bản. Giả thuyết cần kiểm tra: mô hình Instruct với đúng chat format sẽ làm theo chỉ dẫn tốt hơn trong pipeline này. Chưa kết luận đây là nguyên nhân duy nhất; retrieval thiếu bằng chứng và giới hạn context vẫn phải kiểm tra riêng.

## Thiết kế kiểm soát

Thêm Qwen/Qwen2.5-3B-Instruct (model card ghi 3.09B, có tiếng Việt). Runtime vẫn đếm và từ chối model >=4 tỷ tham số. Model mới dùng FP16/eager trên CUDA; chưa đo VRAM thực tế trên P100 cho nhánh này. Encoder, index, split, evaluator và ngân sách 4096/256 giữ nguyên.

| Cấu hình | Retriever | Prompt/packing | Mô hình và định dạng |
|---|---|---|---|
| b0_code | Dense L2 | baseline/prefix | Qwen1.5-1.8B, plain completion |
| b1_cosine | Dense cosine | baseline/prefix | Như B0 |
| i0_dense_chat | Dense cosine | baseline/prefix | Qwen2.5-3B-Instruct, tokenizer chat template |
| i1_bm25_chat | BM25 | baseline/prefix | Như I0 |
| i2_bm25_grounded_chat | BM25 | grounded/balanced | Như I0 |

- B1 → I0 thay **nhóm mô hình sinh + định dạng chat**, không quy toàn bộ chênh lệch cho retrieval hoặc riêng một thành phần của nhóm này.
- I0 → I1 giữ cùng mô hình, định dạng và prompt, chỉ thay retriever.
- I1 → I2 thay nhóm prompt/packing, không quy riêng cho từng thành phần.
- Dùng đủ dev để chọn cấu hình. Ba câu pilot chỉ kiểm tra vận hành và lỗi lặp, chưa đủ chọn hệ thống cuối.

## Cách gọi và bảo toàn dữ liệu

`chat_template=true` là lựa chọn tường minh trong cấu hình mới; mọi cấu hình cũ mặc định false. Token budget xét cả token của chat wrapper. Output chỉ decode phần completion; giữ danh sách EOS của model chat để kết thúc lượt đúng định dạng. Không thêm đáp án RES vào message, không thêm few-shot lấy từ test.

Khóa thêm revision bằng `python3 run.py lock-models --model Qwen/Qwen2.5-3B-Instruct`; revision cũ không bị thay. Không cần xây lại corpus, BM25 hoặc embedding SBERT. Mọi run mới phải dùng thư mục mới: code fingerprint đã thay đổi nên không resume vào run cũ. Vẫn đọc/chấm được predictions cũ; không xóa các mẫu OOM hoặc câu trả lời kém để làm đẹp kết quả.

Người học yêu cầu dùng trực tiếp Docker hiện có, chỉ cài dependency còn thiếu. Hiện chưa yêu cầu cài lại thư viện. GPU 1 đang có dịch vụ khác dùng VRAM; nhánh pilot tiếp theo dùng `CUDA_VISIBLE_DEVICES=3`. Các lệnh cần mạng phải truyền `http_proxy`/`https_proxy` đã cấu hình.

## Nguồn chính thức

- Base model và hướng dẫn sử dụng: https://huggingface.co/Qwen/Qwen1.5-1.8B
- Instruct model, số tham số, ngôn ngữ và quickstart: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Chat templates tương ứng Transformers 4.51.3: https://huggingface.co/docs/transformers/v4.51.3/chat_templating
