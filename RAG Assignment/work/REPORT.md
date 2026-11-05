# Báo cáo RAG hỏi đáp pháp luật

**Trạng thái: CHƯA CÓ BENCHMARK GENERATION ĐỦ ĐIỀU KIỆN NỘP.**

Báo cáo này tổng hợp bằng chứng đã lưu trong thư mục dự án. Nội dung chưa chạy được ghi rõ, không suy diễn kết quả từ thiết kế kỹ thuật. Bài toán là đánh giá trên corpus được cấp; các câu trả lời không mặc nhiên phản ánh pháp luật hiện hành.

## 1 Mục tiêu và yêu cầu

Xây dựng hệ thống RAG cải thiện retrieval và tích hợp context so với baseline. Theo đề: LLM dưới 4 tỷ tham số; ít nhất 100 cặp hỏi–đáp; rubric gồm cải thiện 3 điểm, demo 2 điểm và báo cáo 5 điểm. Ma trận truy vết ở `docs/requirements.md`.

## 2 Khám phá dữ liệu

| Đặc điểm | Giá trị đo |
| --- | --- |
| Tệp văn bản | 16440 |
| Cặp hỏi–đáp | 2745 |
| Số hiệu văn bản tham chiếu phân biệt, parse được | 312 |
| Câu thiếu nhãn số hiệu parse được | 779 |
| Tệp có nội dung trùng byte (số bản dư) | 0 |
| Độ dài văn bản trung vị (ký tự) | 12270.0 |
| Độ dài câu hỏi trung vị (ký tự) | 137 |
| Độ dài đáp án trung vị (ký tự) | 1426 |

| Tập | Số câu | Có ít nhất một số hiệu khớp tên tệp corpus |
| --- | --- | --- |
| test | 100 | 0 |
| dev | 107 | 0 |
| unused | 2487 | 0 |
| diagnostic_dev | 29 | 29 |
| diagnostic_test | 22 | 22 |

Coverage ở đây chỉ là đối chiếu số hiệu với tên tệp sau chuẩn hóa. Không khớp **không chứng minh** đáp án hoàn toàn vắng trong nội dung: văn bản khác có thể trích dẫn quy định. Khớp cũng không đảm bảo đúng cơ quan, địa phương, phiên bản hoặc đoạn chứa đáp án. Không tự tải thêm corpus hoặc sửa nhãn từ nguồn ngoài để làm đẹp benchmark.

Nhãn không trích xuất được được xem là thiếu, không gom thành một nhóm văn bản. Split chính dùng connected components của số hiệu văn bản để giảm giao nhau giữa dev/test. Với câu thiếu nhãn, không thể bảo đảm độc lập theo văn bản chưa biết. Tập diagnostic_dev/diagnostic_test được lấy riêng từ các nhóm còn lại có tên tệp khớp; không thay thế test chính 100 câu.

Độ khó được mô tả qua độ dài đáp án, câu nhiều yêu cầu và thiếu văn bản; chưa có nhãn độ khó do chuyên gia xác nhận. Hash tệp nguồn và thống kê đầy đủ nằm trong `data/audit.json`.

## 3 Baseline và phương pháp

| Mã | Retrieval | Context và prompt | Mục đích |
| --- | --- | --- | --- |
| B0 | SBERT + FAISS squared-L2 | Prompt gốc; cắt prefix theo ngân sách | Tái lập hành vi code sau sửa vận hành |
| B1 | SBERT normalize + inner product (cosine) | Như B0 | Đối chiếu mô tả cosine trong đề |
| R1 | BM25 + dense cosine; RRF | Như B0 | Tách đóng góp hybrid retrieval |
| R2 | Như R1 | Phân bổ token cho các nguồn; prompt có nhãn | Tách context/prompt như một nhóm thay đổi |
| R3 tùy chọn | Hybrid trên chunk theo Điều | Như R2 | Thử cấu trúc pháp luật nếu đủ tài nguyên |
| I0 | Dense cosine như B1 | baseline/prefix; tokenizer chat template | Thay nhóm mô hình sinh + format bằng Qwen2.5-3B-Instruct |
| I1 | BM25 | Như I0 | Tách tác dụng retrieval với cùng Instruct model |
| I2 | BM25 | grounded/balanced; cùng chat template | Tách nhóm prompt/context trên Instruct model |

B0/B1/R1/R2/R3 giữ cùng Qwen/Qwen1.5-1.8B. Nhánh I0/I1/I2 dùng Qwen2.5-3B-Instruct. Tất cả dùng greedy decoding, 4096 input token (kể cả chat wrapper nếu có) và tối đa 256 token sinh. Giới hạn input là chính sách vận hành được công bố; bản gốc không kiểm soát ngân sách tương đương. R2 thay đồng thời packing và prompt nên kết quả chỉ quy cho nhóm thay đổi đó, không riêng từng thành phần. B1 → I0 thay nhóm mô hình sinh + chat format, không nhận là đóng góp retrieval. I0 → I1 giữ cùng generator, format và prompt; I1 → I2 thay nhóm prompt/packing. Mô hình Instruct là giả thuyết cải thiện sau pilot base lặp nội dung, chưa được coi là hiệu quả nếu chưa đo. Chi tiết và nguồn chính thức ở `docs/instruct_experiment.md`.

Hybrid dùng BM25 positive-IDF (k1=1.5, b=0.75), top-30 mỗi retriever, RRF với hằng số 60 và lấy top-5. Đây là cấu hình khai báo trước, chưa có bằng chứng tối ưu. Metadata nguồn không lấy từ nhãn câu hỏi. Mã không yêu cầu fine-tuning; mục tiêu đầu tiên là cải thiện pipeline đúng trọng tâm đề.

## 4 Thiết kế đánh giá

Chọn hệ thống trên dev theo ROUGE-L; phá hòa bằng độ trễ trung vị rồi tên cấu hình. Chỉ sau khi lưu selection mới chạy test chính. Nhãn và đáp án nằm riêng; inference không đọc chúng. Các phép đo giữ cả mẫu lỗi với điểm 0. Model revisions được khóa trong `models.lock.json`.

Metric chính tương thích code baseline: cosine SBERT, Jaccard theo tập, Token Overlap theo Counter, BLEU unigram đơn giản có brevity penalty, ROUGE-L bằng LCS. Xuất thêm Token_Overlap_Set_PDF và BLEU_Set_PDF do công thức tập hợp trong đề khác code. BLEU_Score không phải corpus BLEU-4 chuẩn. Cosine dùng evaluator cố định, không đổi theo retriever. Vietnamese-SBERT giới hạn 256 token nên câu trả lời dài bị cắt khi chấm cosine; token metric xét toàn chuỗi.

Retrieval dùng Hit@5, Recall@5 và MRR@5 ở mức số hiệu văn bản, báo cả toàn tập và phần có nhãn khớp corpus. Không diễn giải metric này thành độ chính xác pháp lý hoặc relevance ở mức passage. Khoảng tin cậy bootstrap ghép cặp dùng câu hỏi làm đơn vị; chưa hiệu chỉnh tương quan giữa nhiều câu cùng văn bản.

## 5 Kết quả thực nghiệm

**Chưa chạy generation đầy đủ trên P100. Chưa có điểm baseline/cải tiến để kết luận hiệu quả hoặc đáp ứng yêu cầu tăng metric.**

### Chẩn đoán retrieval đã chạy

| Tập | Cấu hình | N | N có nhãn khớp | Hit@5 covered | MRR@5 covered |
| --- | --- | --- | --- | --- | --- |
| diagnostic_dev | sparse_diagnostic | 29 | 29 | 0.7931034482758621 | 0.607471264367816 |

Đây là phép đo retrieval, chưa phải 5 metric câu trả lời của assignment.

## 6 Phân tích lỗi và giới hạn

Cần kiểm tra thủ công các câu tăng nhiều, giảm nhiều, trả lời rỗng, hết ngân sách sinh và có nhãn nguồn không hợp lệ. Các ID tăng/giảm được xuất tự động trong `artifacts/comparisons/*/error_analysis_cases.json`; mở predictions và scores để đối chiếu nội dung. Chưa gán nhãn đúng pháp lý bằng LLM tự chấm. Kiểm tra nhãn [S#] chỉ xác minh định dạng và nguồn có được đưa vào prompt, không chứng minh lập luận được nguồn hỗ trợ.

Các giới hạn chính: độ phủ tên tệp thấp; nhiều nhãn thiếu; số hiệu địa phương có thể không duy nhất; encoder cắt 256 token; LLM base và ngân sách output 256 token; query đơn và thời gian gồm cold-start chưa chuẩn hóa hoàn toàn. RSS quan sát tại biên câu hỏi không phải peak RSS toàn tiến trình. GPU peak allocated không bao gồm mọi bộ nhớ của driver/FAISS. Không hứa hẹn cải thiện trước khi chạy.

## 7 Tái lập và demo

Các lệnh chạy, môi trường P100 và cách resume ở `README.md` và `SERVER_RUN.md`. Dùng `python run_all.py --stage pilot --device cuda` kiểm tra trước, sau đó `python run_all.py --stage full --device cuda`. Có thể bật `--include-legal` trước vòng dev đầu tiên. Kịch bản demo ở `DEMO.md`, câu hỏi bảo vệ ở `DEFENSE.md`.

## 8 Nguồn tham khảo

- Đề và notebook baseline được cấp, giữ nguyên trong thư mục cha.
- Vietnamese-SBERT: https://huggingface.co/keepitreal/vietnamese-sbert
- Giới hạn encoder: https://huggingface.co/keepitreal/vietnamese-sbert/blob/main/sentence_bert_config.json
- Qwen1.5-1.8B: https://huggingface.co/Qwen/Qwen1.5-1.8B
- Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Chat templates: https://huggingface.co/docs/transformers/v4.51.3/chat_templating
- SentenceTransformer encode: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
- FAISS distance metrics: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
- RRF, Cormack et al. 2009: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Robertson & Zaragoza 2009, BM25: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
