"""Build the pre-experiment report framework without importing old result values."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Twips
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REF = WORK / 'review_20261116/template_work/reference_exercises.docx'
OUT = ROOT / 'Manual/01_Khung_bao_cao_RAG.docx'
REFERENCE_SHA = '78ffc994ad0cbaa1beaa342ea4dc0570189699701f169bdf0eb2390e5a0252d0'
assert sha256(REF.read_bytes()).hexdigest() == REFERENCE_SHA
assert (ROOT / 'artifact.md').is_file()
OUT.parent.mkdir(parents=True, exist_ok=True)
(ROOT / 'qa').mkdir(exist_ok=True)

def replace_text(el, text):
    first = next(iter(el.iter(qn('w:r'))), None)
    props = first.find(qn('w:rPr')) if first is not None else None
    props = deepcopy(props) if props is not None else None
    for child in list(el):
        if child.tag != qn('w:pPr'):
            el.remove(child)
    r = OxmlElement('w:r')
    if props is not None:
        r.append(props)
    for i, line in enumerate(text.split('\n')):
        if i:
            r.append(OxmlElement('w:br'))
        t = OxmlElement('w:t')
        t.text = line
        r.append(t)
    el.append(r)

doc = Document(REF)
body = doc._element.body
children = list(body)
cover, pagebreak = children[:21], deepcopy(children[21])
sect = deepcopy(body.sectPr)
for child in list(body):
    body.remove(child)
for child in cover:
    body.append(child)
slots = {
    6: 'BÁO CÁO RAG ASSIGNMENT',
    7: 'HỆ THỐNG HỎI ĐÁP PHÁP LUẬT\nVỚI RETRIEVAL AUGMENTED GENERATION',
    10: 'Học phần: Large Language Models', 11: '', 12: 'Học kỳ: HK261',
    14: 'Học viên: Ngô Duy Bảo', 15: 'Mã số học viên: 2570395',
    16: 'Giảng viên: PGS.TS. Quản Thành Thơ', 19: '', 20: 'Tháng 11 năm 2026',
}
for i, text in slots.items():
    replace_text(cover[i], text)
ppr = cover[6].get_or_add_pPr()
ps = ppr.find(qn('w:pStyle'))
if ps is None:
    ps = OxmlElement('w:pStyle'); ppr.insert(0, ps)
ps.set(qn('w:val'), 'Title')
body.append(pagebreak)
body.append(sect)

def font(run, size=None):
    run.font.name = 'Times New Roman'
    if size is not None:
        run.font.size = Pt(size)
    rp = run._r.get_or_add_rPr()
    fonts = rp.find(qn('w:rFonts'))
    for key in list(fonts.attrib):
        if 'theme' in key.lower():
            del fonts.attrib[key]
    for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
        fonts.set(qn('w:' + key), 'Times New Roman')

for name, indent in [('TOC 1', 0), ('TOC 2', 220)]:
    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.base_style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 1.08
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.left_indent = Twips(indent)
    # A right-aligned tab with a dotted leader keeps page numbers aligned.
    tabs = OxmlElement('w:tabs'); tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'right'); tab.set(qn('w:leader'), 'dot')
    tab.set(qn('w:pos'), '9071'); tabs.append(tab)
    style.element.get_or_add_pPr().append(tabs)

# Add numbering for the report's headings without altering existing definitions.
numbering = doc.part.numbering_part.element
abstract_id = max(int(el.get(qn('w:abstractNumId'))) for el in numbering.findall(qn('w:abstractNum'))) + 1
num_id = max(int(el.get(qn('w:numId'))) for el in numbering.findall(qn('w:num'))) + 1
abstract = OxmlElement('w:abstractNum'); abstract.set(qn('w:abstractNumId'), str(abstract_id))
multi = OxmlElement('w:multiLevelType'); multi.set(qn('w:val'), 'multilevel'); abstract.append(multi)
for level, text, style_id in [(0, '%1', 'Heading1'), (1, '%1.%2', 'Heading2')]:
    lvl = OxmlElement('w:lvl'); lvl.set(qn('w:ilvl'), str(level))
    for name, val in [('start', '1'), ('numFmt', 'decimal'), ('pStyle', style_id), ('lvlText', text), ('suff', 'space')]:
        el = OxmlElement('w:' + name); el.set(qn('w:val'), val); lvl.append(el)
    ppr = OxmlElement('w:pPr'); ind = OxmlElement('w:ind')
    ind.set(qn('w:left'), '0'); ind.set(qn('w:hanging'), '0'); ppr.append(ind); lvl.append(ppr)
    abstract.append(lvl)
numbering.append(abstract)
num = OxmlElement('w:num'); num.set(qn('w:numId'), str(num_id))
aid = OxmlElement('w:abstractNumId'); aid.set(qn('w:val'), str(abstract_id)); num.append(aid); numbering.append(num)

def para(text='', placeholder=False, align=None):
    p = doc.add_paragraph(text)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control = True
    for run in p.runs:
        font(run, 12 if placeholder else 13)
        if placeholder:
            run.italic = True
    return p

def heading(text, level=1, new_page=False, numbered=True):
    p = doc.add_paragraph(text, style='Heading ' + str(level))
    p.paragraph_format.page_break_before = new_page
    p.paragraph_format.keep_with_next = True
    if numbered:
        np = OxmlElement('w:numPr')
        ilvl = OxmlElement('w:ilvl'); ilvl.set(qn('w:val'), str(level - 1)); np.append(ilvl)
        nid = OxmlElement('w:numId'); nid.set(qn('w:val'), str(num_id)); np.append(nid)
        p._p.get_or_add_pPr().append(np)
    return p

def wait(text):
    return para('Chờ ' + text, placeholder=True)

def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    tw = t._tbl.tblPr.find(qn('w:tblW'))
    tw.set(qn('w:type'), 'dxa'); tw.set(qn('w:w'), '9071')
    for col, width in zip(t.columns, widths):
        col.width = Twips(width)
    for i, label in enumerate(headers):
        t.rows[0].cells[i].text = label
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = text
    trpr = t.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement('w:tblHeader'); trpr.append(repeat)
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        el = OxmlElement('w:' + edge)
        for name, val in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            el.set(qn('w:' + name), val)
        borders.append(el)
    t._tbl.tblPr.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge, width in [('top', 80), ('bottom', 80), ('left', 90), ('right', 90)]:
        el = OxmlElement('w:' + edge); el.set(qn('w:w'), str(width)); el.set(qn('w:type'), 'dxa'); margins.append(el)
    t._tbl.tblPr.append(margins)
    for ri, row in enumerate(t.rows):
        keep = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(keep)
        for ci, cell in enumerate(row.cells):
            cell.width = Twips(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri == 0:
                shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), 'E8E8E8'); cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.08; p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                for run in p.runs:
                    font(run, 11); run.bold = ri == 0
    para('').paragraph_format.space_after = Pt(0)
    return t

p = para('MỤC LỤC', align=WD_ALIGN_PARAGRAPH.CENTER)
p.paragraph_format.space_after = Pt(14)
for r in p.runs:
    r.bold = True; font(r, 16)
p = para('')
r = OxmlElement('w:r'); begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin'); r.append(begin); p._p.append(r)
r = OxmlElement('w:r'); instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve'); instr.text = ' TOC \\o "1-2" \\h \\z \\u '; r.append(instr); p._p.append(r)
r = OxmlElement('w:r'); sep = OxmlElement('w:fldChar'); sep.set(qn('w:fldCharType'), 'separate'); r.append(sep); p._p.append(r)
r = OxmlElement('w:r'); end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end'); r.append(end); p._p.append(r)

p = para('TÓM TẮT', align=WD_ALIGN_PARAGRAPH.CENTER)
p.paragraph_format.page_break_before = True
for r in p.runs:
    r.bold = True; font(r, 14)
para('Bài tập nghiên cứu hệ thống hỏi đáp pháp luật sử dụng truy hồi kết hợp sinh câu trả lời. Trong báo cáo, tôi xác định bài toán, phân tích dữ liệu, trình bày baseline từ notebook của giảng viên và đánh giá các thay đổi ở bước truy hồi và tích hợp ngữ cảnh.')
wait('điền phương pháp được lựa chọn, số cặp thực tế, kết quả chính và hạn chế sau khi hoàn thành thực nghiệm.')

heading('Giới thiệu bài toán')
heading('Lĩnh vực và nhiệm vụ', 2)
para('Lĩnh vực của bài toán là hỏi đáp trên văn bản pháp luật bằng tiếng Việt. Nhiệm vụ của hệ thống là tiếp nhận câu hỏi, tìm các đoạn văn bản liên quan trong kho tài liệu được cung cấp và sử dụng chúng làm ngữ cảnh cho mô hình ngôn ngữ sinh câu trả lời [1].')
heading('Đầu vào đầu ra và mục tiêu', 2)
para('Đầu vào khi suy luận gồm câu hỏi bằng ngôn ngữ tự nhiên và kho văn bản pháp luật. Đầu ra là câu trả lời cho câu hỏi đó. Đáp án tham chiếu thuộc dữ liệu đánh giá, không được đưa vào quá trình sinh câu trả lời.')
para('Mục tiêu là cải thiện chất lượng câu trả lời so với baseline bằng cách thay đổi truy hồi hoặc cách tích hợp thông tin đã truy hồi. Kết quả được so sánh trên cùng tập ít nhất 100 cặp hỏi đáp bằng các chỉ số quy định trong đề [1].')

heading('Dữ liệu và bối cảnh sử dụng', new_page=True)
heading('Nguồn và cấu trúc dữ liệu', 2)
para('Theo mô tả đề bài, dữ liệu gồm khoảng 16.440 tệp văn bản pháp luật và khoảng 2.745 cặp câu hỏi với câu trả lời tham chiếu [1]. Kho văn bản phục vụ truy hồi; tập hỏi đáp phục vụ đánh giá. Số lượng sử dụng thực tế và các trường dữ liệu được trình bày sau khi kiểm tra bản dữ liệu đầu vào.')
table(['Thành phần', 'Vai trò', 'Số lượng thực tế'], [
    ['Văn bản pháp luật', 'Nguồn truy hồi', 'Chờ kiểm tra'],
    ['Cặp hỏi đáp', 'Đánh giá câu trả lời', 'Chờ kiểm tra'],
    ['Tập phát triển', 'Lựa chọn cấu hình', 'Chờ xác nhận'],
    ['Tập đánh giá chính', 'So sánh cùng câu hỏi', 'Chờ xác nhận\nTối thiểu 100'],
], [3400, 3300, 2371])
heading('Đặc điểm và những khó khăn của dữ liệu', 2)
wait('điền phân bố độ dài văn bản, câu hỏi và đáp án; số liệu phải kèm cách đo thống nhất. Chỉ phân nhóm chủ đề hoặc độ khó khi có căn cứ và tiêu chí rõ ràng.')
wait('bổ sung ví dụ thực về câu hỏi cần tìm điều khoản, câu hỏi cần tổng hợp và trường hợp thiếu ngữ cảnh. Phân biệt nhận xét từ mẫu đã đọc với thống kê toàn bộ dữ liệu.')
para('Các câu trong tập đánh giá được cố định trước khi chạy và dùng chung cho mọi hệ thống. Khi trình bày thiết kế thực nghiệm, tôi sẽ ghi rõ cách chọn tập, việc sử dụng các câu này trong những lần thực nghiệm trước và phạm vi kết luận có thể rút ra.')

heading('Phương pháp đánh giá', new_page=True)
heading('Các chỉ số theo đề bài', 2)
para('Gọi ŷ là câu trả lời sinh ra và y là đáp án tham chiếu. Wŷ và Wy là các tập token sau chuẩn hóa. Các chỉ số dưới đây được áp dụng cho từng cặp câu trả lời, sau đó lấy trung bình trên cùng tập đánh giá [1].')
table(['Chỉ số', 'Định nghĩa sử dụng'], [
    ['Cosine similarity', 'Tích vô hướng hai vector đáp án chia cho tích chuẩn của chúng.'],
    ['Jaccard similarity', '|Wŷ ∩ Wy| / |Wŷ ∪ Wy|'],
    ['Token Overlap', '|Wŷ ∩ Wy| / |Wy|'],
    ['BLEU unigram theo đề', 'BP × |Wŷ ∩ Wy| / |Wŷ|'],
    ['ROUGE L', 'F1 của precision và recall tính từ dãy con chung dài nhất của hai chuỗi token.'],
], [2900, 6171])
para('Trong BLEU unigram, BP bằng 1 nếu chuỗi dự đoán không ngắn hơn chuỗi tham chiếu; nếu ngắn hơn, BP = exp(1 − |y|/|ŷ|). Các độ dài dùng trong BP là độ dài chuỗi token, khác với số phần tử của tập token. Trường hợp rỗng và cách chuẩn hóa được quy định thống nhất trong bộ chấm.')
heading('Đối chiếu với bộ chấm trong notebook', 2)
para('Các hàm Token Overlap và BLEU trong notebook giảng viên đếm số lần xuất hiện của token [2], trong khi công thức của đề dùng tập hợp token [1]. Báo cáo sẽ tách hai cách tính. Mọi hệ thống được chấm bằng cùng định nghĩa trong từng bảng; không so sánh các giá trị thuộc hai định nghĩa khác nhau.')
wait('điền quy tắc chuẩn hóa, embedding chấm cosine, cách xử lý lỗi và đáp án rỗng đã kiểm chứng. Phân tích giới hạn của các chỉ số; không diễn giải similarity thành tỷ lệ câu trả lời đúng.')

heading('Baseline từ notebook của giảng viên', new_page=True)
heading('Ý tưởng và quy trình xử lý', 2)
para('Theo đề bài, baseline chia văn bản thành các đoạn khoảng 2.000 ký tự với phần chồng lấn 50 ký tự. Mô hình Vietnamese SBERT mã hóa các đoạn để lưu vào FAISS. Với mỗi câu hỏi, hệ thống truy hồi năm đoạn liên quan, ghép chúng vào prompt và gọi Qwen/Qwen1.5-1.8B sinh câu trả lời [1].')
para('Đường chạy baseline được thực hiện từ notebook giảng viên [2]. Các thay đổi để chạy trong Docker, đọc dữ liệu cục bộ và chọn tập đánh giá được ghi riêng trong tài liệu mã nguồn. Những thay đổi có ảnh hưởng tới câu trả lời, nếu có, phải được nêu trong thiết lập thực nghiệm.')
table(['Thành phần', 'Thiết lập theo đề hoặc notebook'], [
    ['Chia đoạn', '2.000 ký tự, chồng lấn 50 ký tự'],
    ['Embedding', 'keepitreal/vietnamese-sbert'],
    ['Truy hồi', 'FAISS, lấy 5 đoạn'],
    ['Mô hình sinh', 'Qwen/Qwen1.5-1.8B'],
    ['Khoảng cách và chuẩn hóa thực tế', 'Chờ xác nhận từ notebook đã chạy'],
], [3500, 5571])
heading('Kết quả và phân tích baseline', 2)
wait('điền số câu đã chạy, số lỗi và năm chỉ số sau khi thực thi notebook. Bảng kết quả chung đặt ở phần thực nghiệm.')
wait('phân tích ưu điểm và hạn chế từ câu trả lời và đoạn truy hồi thực tế. Nếu notebook nguyên gốc dùng L2, trình bày đối chứng đó riêng với nhánh cosine theo mô tả đề; không đổi nhãn hai kết quả.')

heading('Phương pháp đề xuất', new_page=True)
heading('Ý tưởng và các thành phần thay đổi', 2)
para('Các hướng được xem xét gồm truy hồi theo từ khóa bằng BM25, kết hợp truy hồi từ khóa với truy hồi vector và điều chỉnh cách đưa các đoạn truy hồi vào prompt. Mỗi thay đổi cần có giả thuyết cụ thể và được đánh giá so với baseline. Nếu thay mô hình sinh, ảnh hưởng của việc đổi mô hình được trình bày rõ trong phép so sánh.')
table(['Thành phần', 'Nội dung cần trình bày'], [
    ['Embedding và retriever', 'Mô hình, cách xếp hạng, tham số và lý do lựa chọn.'],
    ['Prompt và mô hình sinh', 'Prompt thực tế, mô hình dưới 4 tỷ tham số và cấu hình sinh.'],
    ['Tích hợp ngữ cảnh', 'Cách chọn đoạn, thứ tự, ngân sách token và xử lý thiếu thông tin.'],
], [3300, 5771])
wait('bổ sung sơ đồ phương pháp đã triển khai và mô tả các bước bằng lời; chỉ đưa vào báo cáo chính những cấu hình có bằng chứng thực nghiệm.')
heading('Ưu điểm hạn chế và cách kiểm chứng', 2)
wait('trình bày lý do kỳ vọng từng thay đổi có ích, sau đó đối chiếu với phép đo. Không kết luận BM25, prompt hoặc mô hình mới tốt hơn chỉ dựa trên trực giác.')
wait('điền các đối chứng giúp phân biệt tác động của retriever, mô hình sinh và cách tích hợp context. Khi nhiều yếu tố thay đổi cùng lúc, chỉ kết luận về toàn bộ cấu hình nếu chưa có đối chứng tách riêng.')

heading('Thiết lập thực nghiệm và kết quả', new_page=True)
heading('Điều kiện thực nghiệm', 2)
wait('điền phần cứng thực chạy, phiên bản thư viện và mô hình, số cặp, cách chọn tập, seed, kiểu số, tham số sinh và cấu hình có hiệu lực. Tham số chưa xác nhận không được điền theo dự đoán.')
para('Mọi hệ thống được đánh giá trên cùng danh sách câu hỏi và đáp án tham chiếu. Dữ liệu đầu vào, các cấu hình và bộ chấm được cố định để có thể chạy lại. Nếu một câu gặp lỗi, lỗi phải được ghi nhận và giữ trong tổng số mẫu khi tổng hợp kết quả.')
heading('So sánh định lượng', 2)
para('Bảng kết quả chính sử dụng các công thức trong đề. Các cột phương pháp được đặt tên theo cấu hình đã thực thi; chênh lệch được tính từ đúng đối chứng nêu trong bảng.')
table(['Chỉ số', 'Baseline\nnotebook', 'Baseline\ncosine', 'Phương pháp\nđề xuất', 'Chênh lệch'], [
    ['Số cặp và số lỗi', 'Chờ', 'Chờ', 'Chờ', '—'],
    ['Cosine similarity', 'Chờ', 'Chờ', 'Chờ', 'Chờ'],
    ['Jaccard similarity', 'Chờ', 'Chờ', 'Chờ', 'Chờ'],
    ['Token Overlap', 'Chờ', 'Chờ', 'Chờ', 'Chờ'],
    ['BLEU unigram theo đề', 'Chờ', 'Chờ', 'Chờ', 'Chờ'],
    ['ROUGE L', 'Chờ', 'Chờ', 'Chờ', 'Chờ'],
], [2700, 1550, 1550, 1750, 1521])
wait('xác nhận các nhánh baseline thực tế và ghi rõ đối chứng của cột chênh lệch. Nếu hai nhánh tương đương, giải thích và gộp sau kiểm chứng. Không tự tạo một kết quả cosine khi chưa chạy.')
wait('bổ sung bảng phụ theo hàm chấm notebook, kết quả tập phát triển và các đối chứng đã chạy. Nêu cả chỉ số tăng và giảm; chỉ ghi nhận cải thiện khi có số liệu tính lại từ đầu ra thực.')

heading('Phân tích kết quả và hạn chế', new_page=True)
heading('Phân tích các trường hợp cụ thể', 2)
wait('chọn ví dụ từ tập đánh giá có câu hỏi, đáp án tham chiếu, câu trả lời baseline, câu trả lời đề xuất và đoạn nguồn. Bao gồm trường hợp cải thiện và trường hợp chưa đạt; không chỉ chọn ví dụ thuận lợi.')
wait('bổ sung kết quả demo chạy thật trên một tập câu hỏi nhỏ. Nhận xét độ rõ ràng, mức độ liên quan và thông tin được nguồn hỗ trợ; không dùng nhãn trích dẫn hợp lệ để thay cho kiểm tra nội dung.')
heading('Hạn chế và khả năng chạy lại', 2)
wait('phân tích giới hạn dữ liệu, truy hồi, độ dài ngữ cảnh và đánh giá dựa trên quan sát thực. Ghi nhận lịch sử đã xem tập đánh giá và các khác biệt phát sinh khi chạy lại.')
para('Kiểm tra lại điểm số từ câu trả lời đã lưu xác minh cách tổng hợp kết quả. Khả năng chạy lại toàn bộ thực nghiệm cần được kiểm chứng bằng việc xây dựng chỉ mục, sinh câu trả lời và chấm điểm từ môi trường được hướng dẫn. Hai phép kiểm tra này được ghi nhận riêng.')

heading('Kết luận')
wait('viết sau khi đã kiểm chứng kết quả. Tóm tắt phương pháp, mức thay đổi so với baseline trên đúng tập đánh giá và hạn chế còn lại. Không cam kết hiệu quả ngoài phạm vi dữ liệu được đo.')

heading('Tài liệu tham khảo', new_page=True, numbered=False)
para('[1] Quản Thành Thơ. Retrieval Augmented Generation Assignment — Building a Simple Law Chatbot. Phiên bản 1.0, ngày 9 tháng 11 năm 2025.')
para('[2] Notebook baseline do giảng viên cung cấp kèm RAG Assignment.')
wait('bổ sung tài liệu kỹ thuật thực sự được sử dụng và đã đối chiếu. Không thêm tài liệu chưa đọc để làm đầy danh mục.')
heading('Phụ lục hướng dẫn chạy lại thực nghiệm', numbered=False)
para('Phụ lục trình bày các bước để người đọc chạy baseline, các phương pháp đề xuất và bộ chấm trên cùng dữ liệu. Hướng dẫn phải được kiểm tra từ thư mục làm việc sạch trước khi chốt báo cáo.')
wait('điền yêu cầu môi trường, chuẩn bị dữ liệu, lệnh chạy từng cấu hình, thời gian và tài nguyên đo thực tế, vị trí đầu ra và cách đối chiếu kết quả. Tách rõ lệnh chạy mô hình với lệnh xem hoặc chấm lại đầu ra đã có.')

for sec in doc.sections:
    replace_text(sec.header.paragraphs[0]._p, 'Large Language Models  |  RAG Assignment  |  HK261')
    for r in sec.header.paragraphs[0].runs:
        font(r, 10.5)
    for p in sec.footer.paragraphs:
        for r in p.runs:
            font(r, 10.5)
settings = doc.settings.element
up = settings.find(qn('w:updateFields'))
if up is None:
    up = OxmlElement('w:updateFields'); settings.append(up)
up.set(qn('w:val'), 'true')
doc.core_properties.author = 'Ngô Duy Bảo'
doc.core_properties.last_modified_by = 'Ngô Duy Bảo'
doc.core_properties.title = 'Hệ thống hỏi đáp pháp luật với Retrieval Augmented Generation'
doc.core_properties.subject = 'Khung báo cáo RAG Assignment trước đợt chạy notebook giảng viên'
doc.save(OUT)

# Remove unused assets from the replaced Deep Learning body, keeping the reference unchanged.
with ZipFile(OUT) as z:
    parts = {name: z.read(name) for name in z.namelist()}
rels_name = 'word/_rels/document.xml.rels'
rels = E.fromstring(parts[rels_name])
for rel in list(rels):
    if rel.get('Type', '').rsplit('/', 1)[-1] in ['image', 'customXml', 'hyperlink']:
        rels.remove(rel)
parts[rels_name] = E.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
for name in list(parts):
    if name.startswith(('word/media/', 'customXml/')):
        del parts[name]
ct = E.fromstring(parts['[Content_Types].xml'])
for child in list(ct):
    if child.get('PartName') and child.get('PartName').lstrip('/') not in parts:
        ct.remove(child)
parts['[Content_Types].xml'] = E.tostring(ct, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(REF) as z:
    for name in ['word/fontTable.xml', 'word/theme/theme1.xml']:
        assert parts[name] == z.read(name)
with ZipFile(OUT, 'w', ZIP_DEFLATED) as z:
    for name, content in parts.items():
        z.writestr(name, content)

split_source = WORK / 'data/splits.json'
split = json.loads(split_source.read_text(encoding='utf-8'))
ids = split['ids']['test']
assert len(ids) >= 100 and len(ids) == len(set(ids))
contract = {
    'status': 'framework_created_experiments_not_run',
    'minimum_evaluation_pairs': 100,
    'planned_evaluation_ids': ids,
    'evaluation_ids_source': 'data/splits.json',
    'split_source_sha256': sha256(split_source.read_bytes()).hexdigest(),
    'evaluation_history': 'These IDs were used in previous experiments; do not describe them as an unseen holdout.',
    'baseline_origin': '../Baseline.ipynb',
    'baseline_original_sha256': sha256((WORK.parent / 'Baseline.ipynb').read_bytes()).hexdigest(),
    'baseline_execution': 'Execute a documented Docker-compatible copy of the instructor notebook; do not substitute the previous custom pipeline.',
    'preserve_baseline_logic': ['document preprocessing and splitting', 'embedding and retrieval', 'prompt', 'LLM wrapper call and effective generation settings', 'notebook metric functions'],
    'allowed_documented_operational_changes': ['local paths and local RES loading', 'shared fixed evaluation selection >=100', 'logging and saved outputs', 'locking validated dependencies and model revisions'],
    'changes_requiring_evidence': ['compatibility fixes', 'precision or memory changes', 'decoding changes', 'context truncation', 'completion extraction changes'],
    'distance_resolution': 'Inspect actual notebook index/normalization at runtime. If L2, keep as a separate control and execute an explicit cosine notebook branch following the PDF.',
    'metric_tables': {'primary': 'five PDF metrics; set-based overlap and unigram BLEU', 'secondary': 'five instructor notebook metrics; Counter-based overlap and unigram BLEU'},
    'shared_conditions': ['identical ordered question/reference IDs', 'identical metric definitions per table', 'fixed answer evaluator', 'keep error records in N', 'no reference answers during inference'],
    'environment': {'location': 'user-operated server', 'container': 'baond_llm', 'CUDA_VISIBLE_DEVICES': '3', 'create_python_environment': False},
    'quality_status': {'framework_layout': 'pending_render_review', 'original_notebook_run': 'not_run', 'new_full_comparison': 'not_run', 'clean_model_rerun': 'not_run', 'final_report': 'not_ready'},
    'user_review': 'required_after_QA_QC_before_final_packaging',
    'final_zip_allowed': False,
    'submission_allowed': False,
}
(ROOT / 'experiment_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert sha256(REF.read_bytes()).hexdigest() == REFERENCE_SHA
print(json.dumps({'docx': str(OUT), 'tables': len(doc.tables), 'framework_only': True, 'experiments_run': False}, ensure_ascii=True))
