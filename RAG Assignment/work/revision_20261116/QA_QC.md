# QA/QC source code và báo cáo RAG Assignment

## Cách sử dụng

QA là các quy định trước khi chạy để tránh sai yêu cầu; QC là kiểm tra các sản phẩm thực tế sau khi chạy. Các trạng thái hợp lệ: chưa kiểm tra, đang thực hiện, đạt, chưa đạt. Chỉ đánh dấu đạt khi có bằng chứng ở đúng đợt chạy. Việc có script kiểm tra không có nghĩa là script đã chạy thành công.

Mỗi kết luận trong report phải truy về câu hỏi, câu trả lời, đoạn truy hồi, cấu hình, log hoặc phép tính tương ứng. Bảng truy vết kỹ thuật nằm riêng, không đưa đường dẫn, hash hay mã kiểm chứng vào văn phong phần chính của report.

## Ma trận yêu cầu và bằng chứng

| Hạng mục | Điều kiện nghiệm thu | Bằng chứng phải có | Trạng thái đợt mới |
|---|---|---|---|
| Problem Statement | Domain pháp luật; input câu hỏi và kho văn bản; output câu trả lời; mục tiêu cải thiện retrieval và tích hợp context | Mục 1 report khớp mục 1 đề; sơ đồ và đường chạy thật | Chưa kiểm tra bản hoàn chỉnh |
| Khám phá dữ liệu — 1 điểm report | Đếm dữ liệu thực; mô tả cấu trúc, độ dài, sự đa dạng và độ khó có căn cứ; không gọi heuristic tên tệp là ground truth truy hồi | Bảng audit dữ liệu mới, phương pháp đo, ví dụ đã đọc | Chưa kiểm tra |
| Dữ liệu đầu vào | Notebook gốc, ZIP, RES và PDF được bảo toàn; dữ liệu dùng trong thực nghiệm truy được về bản được cấp | SHA256 và kiểm tra cấu trúc, ánh xạ dòng RES sang ID | Chưa kiểm tra đợt mới |
| Tập đánh giá tối thiểu | Có ít nhất 100 cặp ID duy nhất; mọi cấu hình có cùng ID, câu hỏi và đáp án, cùng thứ tự đối chiếu | Danh sách cố định, đối chiếu từng bản ghi, tổng N và lỗi | Chưa kiểm tra |
| Notebook baseline | Thực thi từ notebook của thầy; không thay bằng pipeline tự viết; thay đổi có diff và lý do | Notebook gốc, bản chạy, diff từng cell, notebook đã thực thi, log đầy đủ | Chưa chạy |
| Khoảng cách FAISS | Xác nhận loại khoảng cách, chuẩn hóa và index thực tế; cosine theo đề có cấu hình minh bạch; L2 nếu chạy được ghi riêng | Thông tin index, phép đối chiếu khoảng cách trên vector đã lưu, log từng nhánh | Chưa kiểm tra |
| Logic baseline | Chunk 2000 ký tự, overlap 50, Vietnamese SBERT, top 5, Qwen1.5-1.8B; giữ prompt và cách gọi của notebook trừ thay đổi đã ghi nhận | Cell nguồn và cấu hình thực tế, mẫu prompt/context/output, revisions | Chưa kiểm tra |
| Tương thích môi trường | Không đoán lỗi trước; mọi lỗi thực tế có traceback; thay đổi tối thiểu được kiểm chứng | Log trước/sau sửa, phiên bản thư viện, bản diff, phép chạy thành công | Chưa chạy |
| Giới hạn mô hình | Mỗi LLM dưới 4 tỷ tham số | Tên/revision và số tham số của mô hình đã nạp | Chưa kiểm tra đợt mới |
| Metrics theo đề | Cosine, Jaccard, Token Overlap theo tập hợp, BLEU unigram theo tập hợp và ROUGE-L; chuẩn hóa và tổng hợp thống nhất | Công thức, kiểm tra ví dụ tính tay, điểm từng câu và trung bình tính lại | Chưa kiểm tra |
| Metrics trong notebook | Giữ riêng biến thể đếm lặp của Token Overlap và BLEU; không nhầm với công thức PDF | Chạy hàm notebook và bộ chấm chung trên cùng các chuỗi, bảng đối chiếu | Chưa kiểm tra |
| Công bằng phép so sánh | Cùng tập đánh giá và evaluator; khác biệt LLM/retriever/prompt/context được ghi riêng; mẫu lỗi giữ trong mẫu số | Ma trận cấu hình, tổng số bản ghi, log lỗi và điểm từng câu | Chưa kiểm tra |
| Phương pháp cải tiến — 3 điểm report | Giải thích ý tưởng, lý do lựa chọn, cách triển khai, ưu và nhược điểm; có kết quả lượng hóa | Cấu hình đã thực thi, sơ đồ, prompt thực tế, bảng so sánh và ví dụ | Chưa kiểm tra |
| Cải thiện so với baseline — 3 điểm | Chỉ khẳng định cải thiện khi ít nhất một metric tăng trên cùng tập và cùng định nghĩa; nêu cả metric giảm | Câu trả lời mới, điểm từng câu, bảng chênh lệch tính lại; khoảng tin cậy nếu trình bày | Chưa kiểm tra |
| Lựa chọn phương pháp | Có bản cấu hình cố định trước chạy; công bố việc tập 100 câu đã được xem; không chọn theo test rồi nhận là kết quả holdout độc lập | Thời điểm khóa cấu hình, nhật ký thay đổi, tập phát triển | Chưa kiểm tra |
| Demo — 2 điểm | Chạy thật một tập ví dụ nhỏ; output dễ đọc; kiểm tra tính liên quan và mức độ được nguồn hỗ trợ; giữ ví dụ thất bại | Lệnh/notebook demo, output thực, đoạn nguồn đối chiếu và nhận xét từng ví dụ | Chưa chạy đợt mới |
| Chạy lại thực nghiệm | Người khác làm theo hướng dẫn từ thư mục sạch, xây index, sinh lại câu trả lời và chấm điểm; khác biệt phải được giải thích | Môi trường đã khóa, lệnh thực chạy, log lần chạy lại, so sánh câu trả lời/metrics | Chưa chạy |
| Kiểm tra kết quả đã lưu | Tính lại điểm, đối chiếu số câu và tệp; ghi đúng đây là kiểm tra kết quả lưu | Log chấm lại và kiểm tra tính toàn vẹn | Chưa kiểm tra đợt mới |
| Hình thức — 1 điểm report | Bìa đúng thông tin, có mục lục tự động, bố cục theo nghiên cứu; bảng rõ ràng, có hướng dẫn chạy lại | Word đã cập nhật mục lục, kiểm tra mọi trang và nội dung | Khung đã kiểm tra; chờ bản hoàn chỉnh |
| Nhất quán source và report | Mỗi số liệu/bảng/ví dụ/model/prompt có bằng chứng đúng phiên bản; không có kết quả minh họa giả | Bảng truy vết từng nhận định và kiểm tra số tự động | Chưa kiểm tra bản hoàn chỉnh |
| Review và đóng gói | Bảo đã xem bản đầy đủ và đồng ý chốt trước khi tạo ZIP cuối | Xác nhận của người học; sau đó mới kiểm tra ba thư mục Source code, Dataset, Manual | Chưa review; chưa đóng gói |

## Kiểm tra trước khi chạy GPU

1. Đối chiếu diff với notebook gốc. Xác nhận chỉ có các thay đổi đã kê khai; không cài lại toàn bộ thư viện hoặc đổi thuật toán để tránh một lỗi chưa từng quan sát.
2. Xuất và khóa tập đánh giá có tối thiểu 100 cặp. Xác nhận pipeline suy luận chỉ dùng câu hỏi và corpus, không đọc đáp án để tạo câu trả lời.
3. Kiểm tra corpus, số chunk, embedding, chuẩn hóa và khoảng cách FAISS. Không dùng lại index cũ nếu chưa chứng minh nó đúng với notebook và cấu hình đang chạy.
4. Lưu cấu hình nạp mô hình, revisions, các tham số sinh có hiệu lực và phiên bản thư viện. Kiểm tra ánh xạ GPU vật lý số 3 sang thiết bị logic trong tiến trình.
5. Chạy thử vài câu để phát hiện lỗi vận hành. Pilot không thay cho thực nghiệm tối thiểu 100 cặp. Lưu nguyên output để kiểm tra cách wrapper xử lý prompt và completion.
6. Khi lỗi xảy ra, lưu traceback, cấu hình và thông tin tài nguyên rồi mới sửa. Không hạ giới hạn đầu vào hoặc thay decoding âm thầm để chạy cho xong.

## Kiểm tra sau khi chạy

1. Xác nhận từng phương pháp có đủ N bản ghi, không trùng/thiếu ID, số lỗi công bố và không bỏ mẫu khỏi mẫu số. Chỉ nghiệm thu đường chạy baseline khi các lỗi vận hành đã được xử lý và thực thi lại.
2. Đọc mẫu câu hỏi, context và câu trả lời thực tế trước khi nhìn điểm trung bình. Kiểm tra output lặp, rỗng, lẫn prompt hoặc bị cắt ở giới hạn sinh.
3. Chấm mọi hệ thống bằng cùng evaluator. Đối chiếu các hàm notebook và công thức PDF; không gọi BLEU unigram tự định nghĩa là BLEU chuẩn nhiều bậc.
4. Tính lại bảng và chênh lệch từ điểm từng câu. Nếu báo khoảng tin cậy, lưu phương pháp, số lần lấy mẫu, seed và phép tính để chạy lại.
5. Phân tích ví dụ tốt và chưa tốt bằng đoạn nguồn thật. Nhãn [S1] hợp lệ về cú pháp không chứng minh phát biểu được nguồn hỗ trợ.
6. Phân biệt tăng metric với tỷ lệ câu trả lời đúng; không suy ra hiệu lực pháp luật hiện tại từ điểm similarity hoặc tài liệu lịch sử.
7. Chạy lại từ thư mục sạch theo hướng dẫn TA. So sánh đầu ra và điểm; kết quả khác không giải thích được thì chưa được đánh dấu đạt khả năng tái lập.

## Kiểm tra báo cáo trước khi đưa review

- Đủ domain, task, input/output, dữ liệu và bối cảnh, metrics, baseline, phương pháp đề xuất, so sánh và hạn chế.
- Nêu đúng các kỹ thuật đã thực thi; không nhận đã fine-tune, rerank hoặc chunk theo Điều nếu không có thực nghiệm tương ứng.
- Không điền số cũ vào baseline notebook mới; không dự đoán rằng phương pháp đề xuất chắc chắn thắng.
- Có bằng chứng cho mọi con số và nhận xét. Ý tưởng chưa kiểm chứng phải được gọi là giả thuyết hoặc loại khỏi kết luận.
- Mọi vị trí chờ trong khung đã được điền hoặc xóa có lý do. Không còn hướng dẫn soạn thảo, đường dẫn máy, mã kiểm chứng trong phần chính.
- Cập nhật mục lục, số trang và tài liệu tham khảo. Kiểm tra từng trang Word; chỉ làm DOCX, không xuất báo cáo PDF.
- Đưa toàn bộ source, hướng dẫn và report cho người học review khi QA/QC hoàn tất. Chưa có đồng ý chốt thì chưa tạo ZIP cuối và chưa nộp bài.

## Kết quả kiểm tra bước chuẩn bị ngày 16 tháng 11 năm 2026

- Khung Word: 10 trang đã kiểm tra bằng ảnh từ Word; mục lục tự động có nội dung và số trang. Không nhập số metric cũ, không xuất báo cáo PDF và không tạo ZIP cuối.
- Source notebook: hai nhánh có diff từng cell; kiểm tra AST tám hàm gốc và cú pháp tương thích Python 3.10 đạt.
- Bốn kiểm thử CPU đạt: lời gọi/prompt/raw response giữ nguyên; đúng cùng 100 ID, câu hỏi và đáp án từ RES; từ chối đường dẫn ZIP vượt thư mục; giữ đáp án rỗng trong mẫu số và phân biệt công thức PDF với Counter.
- Hash ba đầu vào gốc đã đối chiếu. Notebook gốc không bị sửa.
- Chưa xác nhận môi trường Docker mới, chạy GPU 100 câu, nhánh cosine thực tế hoặc lần chạy lại mô hình độc lập. Các kiểm tra trên chỉ nghiệm thu bước chuẩn bị, không chứng nhận chất lượng câu trả lời hay mức cải thiện.
