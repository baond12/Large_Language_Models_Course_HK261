# Đợt chỉnh sửa theo yêu cầu đã chốt ngày 16 tháng 11 năm 2026

## Thứ tự thực hiện

Điều chỉnh theo phản hồi mới: chạy thử 1 câu bằng notebook trước, đọc prompt/context/raw response và lỗi thực tế; sau khi đường chạy đạt thì đánh giá đủ 100 câu. Dùng lại chính chỉ mục notebook của lần thử nếu kiểm tra dữ liệu, cấu hình và hash đạt, không xây lại FAISS chỉ vì tăng số câu. Pilot không phải kết quả đánh giá chính.

1. Dựng khung báo cáo Word theo bố cục nghiên cứu đã thống nhất; chuẩn bị bảng đối chiếu yêu cầu và QA/QC.
2. Chuẩn bị bản chạy Docker từ notebook của giảng viên. Giữ nguyên notebook được cấp, ghi rõ từng thay đổi trong bản chạy, kiểm tra khác biệt trước khi thực nghiệm.
3. Người học chạy baseline trên P100 trong Docker và gửi log, notebook đã thực thi cùng kết quả. Xác nhận đủ ít nhất 100 cặp, không bỏ các câu gặp lỗi. Nếu lỗi, lưu traceback trước khi sửa.
4. Chạy các phương pháp đề xuất trên đúng cùng tập đánh giá và cùng bộ chấm điểm. Việc lựa chọn hay điều chỉnh phương pháp dùng tập phát triển; không điều chỉnh theo điểm của tập đánh giá chính.
5. Điền kết quả mới vào báo cáo; viết so sánh, phân tích ví dụ, hạn chế và kết luận theo bằng chứng.
6. QA/QC source, kết quả và báo cáo. Kiểm tra chạy lại thực nghiệm riêng với kiểm tra tính toàn vẹn và chấm lại câu trả lời đã lưu.
7. Đưa bản đầy đủ cho Ngô Duy Bảo review. Chỉ tạo ZIP cuối sau khi người học đồng ý chốt. Không tự nộp bài.

## Trạng thái

- Bản Word trong `Manual` là khung trước thực nghiệm, chưa phải báo cáo hoàn chỉnh.
- Các ô “Chờ kết quả” phải được thay bằng số liệu của đợt chạy notebook giảng viên. Chưa sử dụng các số cũ làm kết quả baseline mới.
- Các phiên bản bài nộp, ZIP, báo cáo, code và kết quả cũ được giữ nguyên để truy vết. Chúng không phải bản đã được người học duyệt cho đợt này.
- Pilot 01 đã lưu FAISS 176.506 vector từ 16.440 văn bản, rồi lỗi duplicate revision. Pilot 02 đã chạy hết 1 câu bằng chỉ mục đó trên P100 số 3, nhưng chuỗi chấm điểm chứa prompt và giới hạn sinh chưa được chuyển đúng qua wrapper. Giữ log/điểm cũ để truy vết, không sử dụng chúng làm điểm câu trả lời của report. Bản sửa định dạng đầu ra và chuyển giới hạn 256 token đã qua kiểm thử CPU, chờ pilot 03. Xem `BASELINE_FAILURE_01.md` và `PILOT_02_OUTPUT_QC.md`.
- Môi trường chạy dự kiến: Docker `baond_llm`, GPU vật lý số 3, `CUDA_VISIBLE_DEVICES=3`; người học tự chạy trên server. Không tạo môi trường Python riêng, không tự nâng cấp toàn bộ thư viện.

## Quy định về baseline

- Baseline mới phải được thực thi từ notebook giảng viên với các hàm truy hồi, prompt, gọi mô hình và chấm điểm của notebook. Không dùng pipeline tự viết trước đây thay thế rồi đổi tên thành baseline notebook.
- Các thay đổi đường dẫn, nạp dữ liệu cục bộ, chọn cùng 100 câu, lưu log và khóa phiên bản phải được ghi rõ. Không tự thay prompt, giới hạn ngữ cảnh, kiểu số, decoding hoặc cách lấy completion dưới nhãn “sửa đường dẫn”.
- Nếu có lỗi tương thích hoặc hết bộ nhớ, giữ log thất bại và thay đổi tối thiểu có giải thích; chạy lại các bước chịu ảnh hưởng. Các thay đổi có thể ảnh hưởng câu trả lời phải được thể hiện trong thiết lập thực nghiệm và phụ lục thay đổi.
- Đề mô tả cosine, còn notebook không đặt tường minh khoảng cách tại chỗ tạo FAISS. Kiểm tra cấu hình và loại khoảng cách thực tế trong môi trường chạy. Nếu bản giữ logic notebook dùng L2, lưu nó như một đối chứng riêng; chuẩn bị thêm nhánh notebook dùng cosine đúng mô tả đề, ghi rõ thay đổi chuẩn hóa/khoảng cách. Không trộn hai kết quả, không âm thầm sửa hoặc khẳng định đã chạy khi chưa có log.
- Bộ chấm của notebook và hai công thức theo tập hợp trong PDF được xuất riêng cho mọi hệ thống. Bảng chính theo đề dùng công thức PDF; bảng phụ ghi kết quả đúng các hàm notebook. Không so chéo hai định nghĩa.

## Tập đánh giá và giới hạn diễn giải

Tập đánh giá dự kiến giữ đúng 100 ID đã có trong `data/splits.json` để so sánh ghép cặp. Trước khi chạy phải xuất danh sách cố định kèm câu hỏi và đáp án từ RES, kiểm tra ID duy nhất, nội dung và dấu vết phiên bản dữ liệu. Cùng một danh sách được dùng cho tất cả hệ thống; không bắt buộc lấy 100 dòng đầu vì đề chỉ yêu cầu tối thiểu 100 cặp.

Tập 100 câu này đã được sử dụng trong các thực nghiệm trước. Vì vậy không được gọi đây là tập chưa từng được xem hoặc xác nhận độc lập. Giữ nguyên các cấu hình ứng viên trước khi chạy lại; mọi thay đổi sau khi xem kết quả phải được ghi nhận. Nếu cần xác nhận độc lập sau điều chỉnh, lập một tập mới và đánh giá mọi hệ thống trên cùng tập đó.

## Tài liệu quản lý

- `Manual/01_Khung_bao_cao_RAG.docx`: cấu trúc và các vị trí điền kết quả.
- `QA_QC.md`: yêu cầu, bằng chứng và điều kiện được phép đánh dấu đạt.
- `Source code/README.md`: các điều chỉnh notebook đã kê khai, bước kiểm tra môi trường và hướng dẫn thực thi sau đó.
- `experiment_contract.json`: ràng buộc cho đợt thực nghiệm; chưa phải chứng nhận đã chạy.
- `qa/`: kiểm tra bố cục và nội dung khung; không phải bằng chứng chất lượng RAG.
