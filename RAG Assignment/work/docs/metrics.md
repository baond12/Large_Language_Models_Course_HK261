# Định nghĩa evaluator

`metrics.py` giữ chuẩn hóa của notebook gốc: lowercase, đổi newline thành space, bỏ ký tự không phải `\w`/whitespace, gom whitespace. Không tự bỏ dấu tiếng Việt, không dùng word segmentation bổ sung cho lexical metric.

| Trường | Định nghĩa |
|---|---|
| Cosine_Similarity | Cosine giữa embedding prediction/reference của Vietnamese-SBERT cùng revision; câu rỗng/lỗi = 0 |
| Jaccard_Similarity | Kích thước giao / hợp của hai tập token |
| Token_Overlap | Tổng min(count_pred,count_ref) / số token reference, giống code baseline |
| BLEU_Score | Unigram precision theo Counter × BP, giống code baseline; không phải BLEU-4 chuẩn |
| ROUGE_L | 2 × LCS_length / (len_pred + len_ref), tương đương F1 của LCS |
| Token_Overlap_Set_PDF | Kích thước giao tập token / kích thước tập reference theo cách viết W trong đề |
| BLEU_Set_PDF | Kích thước giao tập token / kích thước tập prediction × BP theo cách viết W trong đề |

BP = 1 nếu prediction dài ít nhất bằng reference, ngược lại exp(1-len_ref/len_pred). Tất cả trường hợp rỗng chấm 0 theo baseline. Trung bình macro theo câu; không trộn với corpus aggregation khác.

Ví dụ `pred='a a b'`, `ref='a b b'`: Jaccard=1, Counter Token Overlap=2/3, Set Token Overlap=1, BLEU_Score=2/3, ROUGE-L=2/3. Các định nghĩa khác nhau nên phải ghi tên rõ khi báo cáo.

SBERT được khóa cho evaluator độc lập retriever. Giới hạn encoder 256 token khiến cosine có thể chỉ phản ánh đầu đáp án. Không cộng citation vào reference hoặc bỏ citation riêng cho một hệ thống để nâng metric. Đánh giá thủ công khả năng được nguồn hỗ trợ và tính đúng là công việc riêng.

Mỗi câu lỗi inference được giữ với error và câu trả lời rỗng; điểm của nó là 0 trong mọi metric. Không loại lỗi để tính trung bình. Thời gian mỗi câu bao gồm retrieval + context packing + generation; model/index load là chi phí riêng, không nằm trong latency mỗi câu.
