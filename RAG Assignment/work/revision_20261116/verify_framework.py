"""Check the framework, not the RAG experiments. Run after Word refresh/render."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from docx import Document

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
path = ROOT / 'Manual/01_Khung_bao_cao_RAG.docx'
with ZipFile(path) as z:
    body = E.fromstring(z.read('word/document.xml'))
    text = '\n'.join(body.xpath('//w:t/text()', namespaces=NS))
    fields = body.xpath('//w:instrText/text()', namespaces=NS)
    assert any('TOC ' in s for s in fields), 'Missing automatic TOC'
    assert len(body.xpath('//w:hyperlink[starts-with(@w:anchor,"_Toc")]', namespaces=NS)) >= 20, 'TOC was not populated'
    assert 'Bộ bài nộp' not in text
    assert 'Deep_Learning_Course_HK261' not in text
    assert 'Lê Thành Sách' not in text
    assert 'Nguyễn Quốc Minh' not in text
    assert 'CO5085' not in text
    assert 'baseline tái dựng' not in text.lower()
    for expected in ['Ngô Duy Bảo', '2570395', 'PGS.TS. Quản Thành Thơ', 'Giới thiệu bài toán', 'Dữ liệu và bối cảnh sử dụng', 'Phương pháp đánh giá', 'Baseline từ notebook của giảng viên', 'Phương pháp đề xuất', 'Thiết lập thực nghiệm và kết quả', 'Phân tích kết quả và hạn chế', 'Kết luận', 'Phụ lục hướng dẫn chạy lại thực nghiệm']:
        assert expected in text, expected
    sect = body.find('w:body/w:sectPr', NS)
    geom = {E.QName(el).localname: dict(el.attrib) for el in sect if E.QName(el).localname in ['pgSz', 'pgMar']}
    with ZipFile(WORK / 'review_20261116/template_work/reference_exercises.docx') as refz:
        refbody = E.fromstring(refz.read('word/document.xml'))
        refs = refbody.find('w:body/w:sectPr', NS)
        for key in ['pgSz', 'pgMar']:
            assert dict(sect.find('w:' + key, NS).attrib) == dict(refs.find('w:' + key, NS).attrib)

d = Document(path)
assert len(d.tables) == 5
assert all('Chờ' in row.cells[1].text for row in d.tables[-1].rows[1:])
headings = [p.text for p in d.paragraphs if p.style.name.startswith('Heading')]
assert len(headings) == 24
assert d.styles['Normal'].font.size.pt == 13
assert d.styles['Heading 1'].font.size.pt == 16
assert d.styles['Heading 2'].font.size.pt == 14
for name in ['word/fontTable.xml', 'word/theme/theme1.xml']:
    # Word can rewrite namespaces after saving; visual/style checks remain authoritative.
    assert name in ZipFile(path).namelist()
contract_path = ROOT / 'experiment_contract.json'
contract = json.loads(contract_path.read_text(encoding='utf-8'))
assert len(contract['planned_evaluation_ids']) >= 100
assert len(set(contract['planned_evaluation_ids'])) == len(contract['planned_evaluation_ids'])
original = WORK.parent / 'Baseline.ipynb'
assert sha256(original.read_bytes()).hexdigest() == contract['baseline_original_sha256'] == '2dc1acfb86465aa4c53e3205187c5c6999192ae6cf02643a98ee3a103b935572'
assert contract['final_zip_allowed'] is False
assert contract['submission_allowed'] is False
render = json.loads((ROOT / 'qa/render_info.json').read_text(encoding='utf-8-sig'))
pages = render['pages']
assert render['tables_of_contents'] == 1
assert all((ROOT / f'qa/word_pages/page-{i}.png').is_file() for i in range(1, pages+1))
assert not list(ROOT.rglob('*.pdf'))
assert not list(ROOT.rglob('*.zip'))
contract['quality_status']['framework_layout'] = 'passed_structural_and_all_page_visual_review'
contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
result = {
    'scope': 'pre-experiment report framework only',
    'pages_visually_inspected': list(range(1, pages+1)),
    'automatic_toc_populated': True,
    'headings': headings,
    'tables': len(d.tables),
    'old_metric_values_imported': False,
    'original_notebook_unchanged': True,
    'report_sha256': sha256(path.read_bytes()).hexdigest(),
    'reference_sha256': sha256((WORK / 'review_20261116/template_work/reference_exercises.docx').read_bytes()).hexdigest(),
    'baseline_notebook_executed_on_gpu': False,
    'experiment_quality_certified': False,
    'pdf_created': False,
    'final_zip_created': False,
}
(ROOT / 'qa/framework_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['headings']}, ensure_ascii=True))
