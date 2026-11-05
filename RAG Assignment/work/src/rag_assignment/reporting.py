from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .common import read_json, read_jsonl, write_json


def table(headers: list[str], rows: list[list]) -> str:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(escape(v) for v in row) + " |" for row in rows])


def build_report(root: Path) -> dict:
    audit = read_json(root / "data/audit.json")
    summaries = [(p, read_json(p)) for p in sorted((root / "runs").glob("*/*/summary.json"))]
    full = [(p, s) for p, s in summaries if not s["limited_run"]]
    comparisons = [read_json(p) for p in sorted((root / "artifacts/comparisons").glob("*/comparison.json"))]
    status = "ĐÃ CÓ KẾT QUẢ TEST; CẦN KIỂM TRA NỘI DUNG VÀ DEMO" if comparisons else "CHƯA CÓ BENCHMARK GENERATION ĐỦ ĐIỀU KIỆN NỘP"
    selected_path = root / "artifacts/selection.json"
    selection = read_json(selected_path) if selected_path.exists() else None
    lines = ["# Báo cáo RAG hỏi đáp pháp luật", "", f"**Trạng thái: {status}.**", "",
        "Báo cáo này tổng hợp bằng chứng đã lưu trong thư mục dự án. Nội dung chưa chạy được ghi rõ, không suy diễn kết quả từ thiết kế kỹ thuật. "
        "Bài toán là đánh giá trên corpus được cấp; các câu trả lời không mặc nhiên phản ánh pháp luật hiện hành.", "",
        "## 1 Mục tiêu và yêu cầu", "",
        "Xây dựng hệ thống RAG cải thiện retrieval và tích hợp context so với baseline. Theo đề: LLM dưới 4 tỷ tham số; "
        "ít nhất 100 cặp hỏi–đáp; rubric gồm cải thiện 3 điểm, demo 2 điểm và báo cáo 5 điểm. Ma trận truy vết ở `docs/requirements.md`.", "",
        "## 2 Khám phá dữ liệu", "",
        table(["Đặc điểm", "Giá trị đo"], [["Tệp văn bản", audit["documents"]], ["Cặp hỏi–đáp", audit["questions"]],
            ["Số hiệu văn bản tham chiếu phân biệt, parse được", audit["unique_reference_law_ids"]],
            ["Câu thiếu nhãn số hiệu parse được", audit["questions_without_parseable_law_label"]],
            ["Tệp có nội dung trùng byte (số bản dư)", audit["exact_duplicate_documents"]],
            ["Độ dài văn bản trung vị (ký tự)", audit["document_characters"]["median"]],
            ["Độ dài câu hỏi trung vị (ký tự)", audit["question_characters"]["median"]],
            ["Độ dài đáp án trung vị (ký tự)", audit["answer_characters"]["median"]]]), "",
        table(["Tập", "Số câu", "Có ít nhất một số hiệu khớp tên tệp corpus"],
              [[s, v["questions"], v["any_reference_in_corpus"]] for s, v in audit["reference_coverage"].items()]), "",
        "Coverage ở đây chỉ là đối chiếu số hiệu với tên tệp sau chuẩn hóa. Không khớp **không chứng minh** đáp án hoàn toàn vắng trong nội dung: "
        "văn bản khác có thể trích dẫn quy định. Khớp cũng không đảm bảo đúng cơ quan, địa phương, phiên bản hoặc đoạn chứa đáp án. "
        "Không tự tải thêm corpus hoặc sửa nhãn từ nguồn ngoài để làm đẹp benchmark.", "",
        "Nhãn không trích xuất được được xem là thiếu, không gom thành một nhóm văn bản. Split chính dùng connected components "
        "của số hiệu văn bản để giảm giao nhau giữa dev/test. Với câu thiếu nhãn, không thể bảo đảm độc lập theo văn bản chưa biết. "
        "Tập diagnostic_dev/diagnostic_test được lấy riêng từ các nhóm còn lại có tên tệp khớp; không thay thế test chính 100 câu.", "",
        "Độ khó được mô tả qua độ dài đáp án, câu nhiều yêu cầu và thiếu văn bản; chưa có nhãn độ khó do chuyên gia xác nhận. "
        "Hash tệp nguồn và thống kê đầy đủ nằm trong `data/audit.json`.", "",
        "## 3 Baseline và phương pháp", "",
        table(["Mã", "Retrieval", "Context và prompt", "Mục đích"], [
            ["B0", "SBERT + FAISS squared-L2", "Prompt gốc; cắt prefix theo ngân sách", "Tái lập hành vi code sau sửa vận hành"],
            ["B1", "SBERT normalize + inner product (cosine)", "Như B0", "Đối chiếu mô tả cosine trong đề"],
            ["R1", "BM25 + dense cosine; RRF", "Như B0", "Tách đóng góp hybrid retrieval"],
            ["R2", "Như R1", "Phân bổ token cho các nguồn; prompt có nhãn", "Tách context/prompt như một nhóm thay đổi"],
            ["R3 tùy chọn", "Hybrid trên chunk theo Điều", "Như R2", "Thử cấu trúc pháp luật nếu đủ tài nguyên"],
            ["I0", "Dense cosine như B1", "baseline/prefix; tokenizer chat template", "Thay nhóm mô hình sinh + format bằng Qwen2.5-3B-Instruct"],
            ["I1", "BM25", "Như I0", "Tách tác dụng retrieval với cùng Instruct model"],
            ["I2", "BM25", "grounded/balanced; cùng chat template", "Tách nhóm prompt/context trên Instruct model"]]), "",
        "B0/B1/R1/R2/R3 giữ cùng Qwen/Qwen1.5-1.8B. Nhánh I0/I1/I2 dùng Qwen2.5-3B-Instruct. "
        "Tất cả dùng greedy decoding, 4096 input token (kể cả chat wrapper nếu có) và tối đa 256 token sinh. "
        "Giới hạn input là chính sách vận hành được công bố; bản gốc không kiểm soát ngân sách tương đương. "
        "R2 thay đồng thời packing và prompt nên kết quả chỉ quy cho nhóm thay đổi đó, không riêng từng thành phần. "
        "B1 → I0 thay nhóm mô hình sinh + chat format, không nhận là đóng góp retrieval. I0 → I1 giữ cùng generator, format và prompt; "
        "I1 → I2 thay nhóm prompt/packing. Mô hình Instruct là giả thuyết cải thiện sau pilot base lặp nội dung, chưa được coi là hiệu quả nếu chưa đo. "
        "Chi tiết và nguồn chính thức ở `docs/instruct_experiment.md`.", "",
        "Hybrid dùng BM25 positive-IDF (k1=1.5, b=0.75), top-30 mỗi retriever, RRF với hằng số 60 và lấy top-5. "
        "Đây là cấu hình khai báo trước, chưa có bằng chứng tối ưu. Metadata nguồn không lấy từ nhãn câu hỏi. "
        "Mã không yêu cầu fine-tuning; mục tiêu đầu tiên là cải thiện pipeline đúng trọng tâm đề.", "",
        "## 4 Thiết kế đánh giá", "",
        "Chọn hệ thống trên dev theo ROUGE-L; phá hòa bằng độ trễ trung vị rồi tên cấu hình. "
        "Chỉ sau khi lưu selection mới chạy test chính. Nhãn và đáp án nằm riêng; inference không đọc chúng. "
        "Các phép đo giữ cả mẫu lỗi với điểm 0. Model revisions được khóa trong `models.lock.json`.", "",
        "Metric chính tương thích code baseline: cosine SBERT, Jaccard theo tập, Token Overlap theo Counter, BLEU unigram đơn giản có brevity penalty, "
        "ROUGE-L bằng LCS. Xuất thêm Token_Overlap_Set_PDF và BLEU_Set_PDF do công thức tập hợp trong đề khác code. "
        "BLEU_Score không phải corpus BLEU-4 chuẩn. Cosine dùng evaluator cố định, không đổi theo retriever. "
        "Vietnamese-SBERT giới hạn 256 token nên câu trả lời dài bị cắt khi chấm cosine; token metric xét toàn chuỗi.", "",
        "Retrieval dùng Hit@5, Recall@5 và MRR@5 ở mức số hiệu văn bản, báo cả toàn tập và phần có nhãn khớp corpus. "
        "Không diễn giải metric này thành độ chính xác pháp lý hoặc relevance ở mức passage. Khoảng tin cậy bootstrap ghép cặp dùng câu hỏi làm đơn vị; "
        "chưa hiệu chỉnh tương quan giữa nhiều câu cùng văn bản.", "",
        "## 5 Kết quả thực nghiệm", ""]
    if full:
        keys = ["Cosine_Similarity", "Jaccard_Similarity", "Token_Overlap", "BLEU_Score", "ROUGE_L"]
        lines += [table(["Tập", "Hệ thống", "N", "Lỗi", *keys, "Median s/câu"],
            [[s["split"], s["name"], s["questions"], s["errors"], *[f"{s['metrics'][k]:.6f}" for k in keys],
              f"{s['latency_seconds']['median']:.3f}"] for _, s in full]), ""]
    else:
        lines += ["**Chưa chạy generation đầy đủ trên P100. Chưa có điểm baseline/cải tiến để kết luận hiệu quả hoặc đáp ứng yêu cầu tăng metric.**", ""]
    if any(s["limited_run"] for _, s in summaries):
        lines += ["Có kết quả pilot ít mẫu trong `runs/pilot/`; chỉ dùng kiểm tra vận hành, không dùng làm benchmark đủ 100 câu.", ""]
    retrieval_summaries = [read_json(p) for p in sorted((root / "artifacts").glob("retrieval_*/*/summary.json"))]
    if retrieval_summaries:
        lines += ["### Chẩn đoán retrieval đã chạy", "", table(["Tập", "Cấu hình", "N", "N có nhãn khớp", "Hit@5 covered", "MRR@5 covered"],
            [[s["split"], s["config"]["name"], s["questions"], s["reference_covered_questions"],
              s["covered_questions_metrics"].get("Hit@5"), s["covered_questions_metrics"].get("MRR@5")] for s in retrieval_summaries]), "",
            "Đây là phép đo retrieval, chưa phải 5 metric câu trả lời của assignment.", ""]
    if selection:
        lines += [f"Cấu hình được chọn trên dev: `{selection['selected_config']['name']}`. Chi tiết và ID hash ở `artifacts/selection.json`.", ""]
    if comparisons:
        lines += ["### So sánh ghép cặp trên test", ""]
        for c in comparisons:
            lines += [f"{c['baseline']['name']} → {c['candidate']['name']}, cùng {c['paired_questions']} câu.", "",
                table(["Metric", "Delta", "CI95 thấp", "CI95 cao"], [[k, f"{d['mean_delta']:.6f}", f"{d['ci95_low']:.6f}", f"{d['ci95_high']:.6f}"] for k, d in c["deltas"].items()]), ""]
    lines += ["## 6 Phân tích lỗi và giới hạn", "",
        "Cần kiểm tra thủ công các câu tăng nhiều, giảm nhiều, trả lời rỗng, hết ngân sách sinh và có nhãn nguồn không hợp lệ. "
        "Các ID tăng/giảm được xuất tự động trong `artifacts/comparisons/*/error_analysis_cases.json`; "
        "mở predictions và scores để đối chiếu nội dung. Chưa gán nhãn đúng pháp lý bằng LLM tự chấm. "
        "Kiểm tra nhãn [S#] chỉ xác minh định dạng và nguồn có được đưa vào prompt, không chứng minh lập luận được nguồn hỗ trợ.", "",
        "Các giới hạn chính: độ phủ tên tệp thấp; nhiều nhãn thiếu; số hiệu địa phương có thể không duy nhất; encoder cắt 256 token; "
        "LLM base và ngân sách output 256 token; query đơn và thời gian gồm cold-start chưa chuẩn hóa hoàn toàn. "
        "RSS quan sát tại biên câu hỏi không phải peak RSS toàn tiến trình. GPU peak allocated không bao gồm mọi bộ nhớ của driver/FAISS. "
        "Không hứa hẹn cải thiện trước khi chạy.", "",
        "## 7 Tái lập và demo", "",
        "Các lệnh chạy, môi trường P100 và cách resume ở `README.md` và `SERVER_RUN.md`. "
        "Dùng `python run_all.py --stage pilot --device cuda` kiểm tra trước, sau đó `python run_all.py --stage full --device cuda`. "
        "Có thể bật `--include-legal` trước vòng dev đầu tiên. Kịch bản demo ở `DEMO.md`, câu hỏi bảo vệ ở `DEFENSE.md`.", "",
        "## 8 Nguồn tham khảo", "",
        "- Đề và notebook baseline được cấp, giữ nguyên trong thư mục cha.",
        "- Vietnamese-SBERT: https://huggingface.co/keepitreal/vietnamese-sbert", 
        "- Giới hạn encoder: https://huggingface.co/keepitreal/vietnamese-sbert/blob/main/sentence_bert_config.json",
        "- Qwen1.5-1.8B: https://huggingface.co/Qwen/Qwen1.5-1.8B",
        "- Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
        "- Chat templates: https://huggingface.co/docs/transformers/v4.51.3/chat_templating",
        "- SentenceTransformer encode: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html",
        "- FAISS distance metrics: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances",
        "- RRF, Cormack et al. 2009: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
        "- Robertson & Zaragoza 2009, BM25: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    checks = {"data_audit": "done", "primary_test_questions": audit["split_counts"]["test"],
              "primary_test_at_least_100": audit["split_counts"]["test"] >= 100,
              "model_parameter_constraint": "verified in inference manifest" if summaries else "pending actual model loading",
              "generation_comparison": "available" if comparisons else "pending P100 execution",
              "shows_metric_gain": any(c["shows_any_primary_metric_gain"] for c in comparisons) if comparisons else None,
              "demo_content_review": "pending manual review", "report_status": status}
    write_json(root / "artifacts/checklist.json", checks)
    return {"report": str(root / "REPORT.md"), "checklist": checks}


def make_package(root: Path, kind: str = "results") -> dict:
    output = root / "dist" / f"rag_assignment_{kind}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind == "source":
        candidates = [p for folder in ["src", "configs", "tests", "docs", "scripts"] for p in (root / folder).rglob("*") if p.is_file()]
        candidates += [p for p in root.iterdir() if p.is_file() and (p.suffix in {".py", ".md", ".txt", ".json", ".ipynb", ".toml"}
                      or p.name in {"Dockerfile", ".dockerignore", ".gitignore"})]
    else:
        candidates = [p for folder in ["artifacts", "runs"] for p in (root / folder).rglob("*") if p.is_file()]
        candidates += [root / name for name in ["REPORT.md", "models.lock.json", "data/audit.json", "data/manifest.json", "data/splits.json"]]
    files = sorted({p for p in candidates if p.exists() and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    return {"path": str(output), "files": len(files), "bytes": output.stat().st_size,
            "contains_models_or_full_corpus": False}
