# Audit baseline gốc

Nguồn: `../Baseline.ipynb`, các cell được đánh số từ 0. Không sửa notebook gốc.

| Quan sát | Vị trí | Cách xử lý | Phân loại |
|---|---|---|---|
| Giải nén vào `/content/` nhưng điều kiện kiểm tra `/content/dataset`; corpus thực tế ở `export_1/` | Cell 2–3, ZIP | Đọc trực tiếp ZIP hoặc chuẩn hóa root bằng cấu hình; không phụ thuộc Drive | Sửa vận hành |
| Tải RES từ Google Sheets thay vì bản XLSX đã cấp | Cell 2 | Dùng bản RES.xlsx, lưu hash; không cần kết nối sheet | Sửa vận hành/tái lập |
| `df.head(30)` trái tối thiểu 100 trong đề | Cell 5 | Lưu split dev/test; test ít nhất 100 cho tất cả hệ thống | Tuân thủ đề, không phải cải tiến |
| FAISS.from_texts không chỉ định distance/normalization; không có Normalize module trong Vietnamese-SBERT | Cell 3, HF modules.json | Tái lập IndexFlatL2 không normalize làm B0; thêm B1 cosine riêng | Khác biệt code/đề, phải báo |
| Tính batch_embeddings trước, rồi API from_texts/add_texts có thể encode lại | Cell 3 | Encode đúng một lần, cache vector, xây FAISS trực tiếp | Tối ưu vận hành, không nhận là tăng chất lượng |
| Truyền chuỗi text vào FAISS không truyền metadata | Cell 3–4; output metadata={} | Giữ source filename, doc/chunk IDs song song vector | Truy vết; không tự đổi nội dung baseline |
| temperature/max_length nằm trong model_kwargs; cấu hình generation không được tách rõ | Cell 5 | Gọi generate với greedy, max_new_tokens rõ ràng | Sửa vận hành; ghi cấu hình |
| Không xác nhận output chỉ chứa completion | Cell 5 | Slice token sau độ dài prompt; test việc prompt/context không lặp vào câu trả lời | Sửa evaluator để công bằng |
| top-5 × 2000 ký tự có thể vượt ngân sách token | Cell 3/5 | Lưu prompt/input token count, chính sách cắt cùng điều kiện; R2 có packing tốt hơn | Ràng buộc vận hành và ablation |
| Token overlap và BLEU đơn giản dùng Counter; đề viết theo tập hợp W | Cell 5 và PDF tr.2 | Giữ metric baseline làm chính cho so sánh, xuất thêm phiên bản set của đề | Khác biệt evaluator, không trộn |
| Cosine dùng cùng biến embeddings của retriever | Cell 5 | Khóa evaluator keepitreal/vietnamese-sbert cho mọi cấu hình | Công bằng nếu thay embedding |
| Bắt exception và chấm câu trả lời rỗng | Cell 5 | Giữ mẫu lỗi với điểm 0 và trường error; công bố số lỗi | Bảo toàn mẫu số |

Baseline tái lập trong dự án được gọi là **baseline đã sửa lỗi vận hành**. Không khẳng định bit-for-bit với output notebook cũ: thư viện, thứ tự tệp gốc, cấu hình decoding và seed cũ chưa đủ xác định. Mọi phép so sánh chính đều chạy lại trong cùng môi trường, trên cùng dữ liệu và evaluator.
