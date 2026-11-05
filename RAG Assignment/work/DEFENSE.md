# Ghi chú bảo vệ bài

## Vì sao RAG phù hợp bài toán

Embedding mã hóa ngữ nghĩa thành vector; retriever chọn bằng chứng ngoài trọng số LLM; generator dùng bằng chứng trong context. RAG không tự loại hết hallucination. Sai retrieval, mất nội dung do truncation hoặc LLM không làm theo nguồn đều có thể tạo câu trả lời sai.

## Vì sao có hai baseline

Đề mô tả cosine, nhưng notebook gọi FAISS mặc định và không normalize. L2 trên vector thô chịu ảnh hưởng norm; cosine so góc sau normalize. B0 phản ánh code, B1 phản ánh mô tả; phải báo riêng để không nhận một sửa khác biệt làm toàn bộ đóng góp phương pháp.

## Hybrid bổ sung gì

BM25 khai thác trùng từ, số hiệu và thuật ngữ. Dense khai thác biểu diễn ngữ nghĩa nhưng có giới hạn encoder và có thể bỏ lỡ ký hiệu. RRF cộng điểm theo thứ hạng thay vì trộn trực tiếp hai thang score. Chỉ kết luận có ích khi dev/test và ablation chứng minh.

## Context packing liên quan gì attention

Mô hình chỉ attention được trên token đã đưa vào input. Truy hồi đúng nhưng bằng chứng bị cắt thì generator vẫn thiếu thông tin. Chia ngân sách cho nhiều nguồn có thể giữ đa dạng bằng chứng, nhưng cũng có thể cắt mất một điều khoản dài; cần phân tích cả hai loại lỗi.

## Vì sao không fine-tune ngay

Đề ưu tiên retrieval và tích hợp context. Giữ cùng LLM giúp tách đóng góp pipeline, tiết kiệm tài nguyên và giảm nguy cơ ghi nhớ đáp án RES. Fine-tuning chỉ là thử nghiệm bổ sung, cần tập train độc lập và ablation riêng.

## Metric có đo được đúng luật không

Không. Lexical overlap chịu ảnh hưởng cách diễn đạt; cosine encoder bị cắt 256 token; câu nói gần đáp án vẫn có thể sai số tiền, thời hạn hoặc ngoại lệ. Phải đọc nguồn, kiểm tra số liệu và điều kiện của các ca demo.

## Vì sao coverage thấp quan trọng

Nếu không có văn bản đúng, nâng retriever không đảm bảo tìm được đáp án. Coverage tên tệp chỉ là nhãn yếu, vẫn phải kiểm tra nội dung. Không được dùng số hiệu lấy từ cột nhãn để dẫn đường cho inference; đó là rò rỉ dữ liệu.

## Tại sao không dùng test để chọn cấu hình

Chọn theo test làm kết quả lạc quan. Dev dùng để quyết định; test chỉ xác nhận sau khi khóa cấu hình. Sau khi đã xem test mà sửa phương pháp, phải nói rõ và dùng holdout mới để xác nhận độc lập.

## Các câu cần tự hoàn thiện sau khi có kết quả

- Thành phần nào tạo tăng/giảm metric, bằng số liệu nào?
- Ca nào retrieval đúng nhưng generation sai và vì sao?
- Chi phí index, median/p95 latency, VRAM thực tế trên P100 là bao nhiêu?
- Vì sao cấu hình cuối đáng chọn so với phương án nhanh/rẻ hơn?
- Phần nào chưa kiểm chứng và cách cải thiện tiếp theo là gì?
