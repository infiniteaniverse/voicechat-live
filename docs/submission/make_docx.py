# Builds BCSE306L_TeamName_FinalReport.docx from report_docx.html (see build_report.py, REPORT_IMG=1).
# Needs pandoc on PATH or in PANDOC, and python-docx.
#   REPORT_IMG=1 python docs/submission/build_report.py
#   python docs/submission/make_docx.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Mm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
PANDOC = os.environ.get("PANDOC", "pandoc")
OUT = HERE / "BCSE306L_TeamName_FinalReport.docx"
RAW = HERE / "_raw.docx"
REF = HERE / "_ref.docx"
LUA = HERE / "_pb.lua"
NAVY = RGBColor(0x14, 0x31, 0x5C)

LUA.write_text(
    """
local pb = pandoc.RawBlock('openxml', '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
function Div(el)
  if el.classes:includes('pagebreak') then return pb end
end
function Header(el)
  if el.classes:includes('pb') then return { pb, el } end
end
""",
    encoding="utf-8",
)

with open(REF, "wb") as f:
    f.write(subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"], check=True, capture_output=True).stdout)

subprocess.run(
    [PANDOC, str(HERE / "report_docx.html"), "-f", "html", "-t", "docx", "--reference-doc", str(REF),
     "--lua-filter", str(LUA), "--resource-path", str(HERE), "-o", str(RAW)],
    check=True,
)

doc = Document(RAW)


def font(style, name="Calibri", size=None, bold=None, color=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


# heading levels: the cover's <h1> duplicates the Title paragraph (drop it); sections h2 -> Heading 1, h3 -> Heading 2
_remap = {"Heading 2": "Heading 1", "Heading 3": "Heading 2"}
_paras = list(doc.paragraphs)
for p in _paras:
    if p.style.name == "Heading 1":
        p._p.getparent().remove(p._p)
for p in _paras:
    if p.style.name in _remap:
        p.style = doc.styles[_remap[p.style.name]]

S = {s.name: s for s in doc.styles}
for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Image Caption", "Table Caption", "Caption"):
    if name in S:
        font(S[name], "Calibri", 10.5 if name in ("Normal", "Body Text", "First Paragraph") else None)
for name in ("Body Text", "First Paragraph"):
    if name in S:
        S[name].paragraph_format.space_before = Pt(0)
        S[name].paragraph_format.space_after = Pt(6)
        S[name].paragraph_format.line_spacing = 1.12
if "Compact" in S:
    font(S["Compact"], "Calibri", 9)
    S["Compact"].paragraph_format.space_after = Pt(1)
    S["Compact"].paragraph_format.space_before = Pt(1)
for name, size, before, after in (("Title", 24, 0, 6), ("Heading 1", 15, 16, 5), ("Heading 2", 12, 10, 3), ("Heading 3", 11, 8, 2)):
    if name in S:
        font(S[name], "Calibri", size, True, NAVY)
        S[name].paragraph_format.space_before = Pt(before)
        S[name].paragraph_format.space_after = Pt(after)
        S[name].paragraph_format.keep_with_next = True
if "Heading 1" in S:  # rule under section headings
    ppr = S["Heading 1"].element.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "1"), ("w:color", "14315C")):
        b.set(qn(k), v)
    bdr.append(b)
    ppr.append(bdr)
for name in ("Source Code", "Verbatim Char"):
    if name in S:
        font(S[name], "Consolas", 8)
if "Source Code" in S:
    ppr = S["Source Code"].element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", "F6F8FA")):
        shd.set(qn(k), v)
    ppr.append(shd)
    S["Source Code"].paragraph_format.space_after = Pt(0)
if "Placeholder" in S:  # fill-in fields on the cover and contributions table
    rpr = S["Placeholder"].element.get_or_add_rPr()
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", "FFF3A8")):
        shd.set(qn(k), v)
    rpr.append(shd)
    S["Placeholder"].font.bold = True

# cover: spacing so the title sits a third of the way down; subtitle in a lighter, larger face
for p in doc.paragraphs[:6]:
    if p.style.name == "Title":
        p.paragraph_format.space_before = Pt(110)
        p.paragraph_format.space_after = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        break
for p in doc.paragraphs[:8]:
    t_ = p.text.strip()
    if t_.startswith("School of Computer Science"):
        p.paragraph_format.space_before = Pt(0)
        for r in p.runs:
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)
    if t_.startswith("Final project report"):
        p.paragraph_format.space_after = Pt(30)
        for r in p.runs:
            r.font.size = Pt(13)
            r.font.color.rgb = RGBColor(0x3D, 0x48, 0x52)

# page: A4, margins, header/footer with page numbers (not on the cover)
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(18)
sec.top_margin = sec.bottom_margin = Mm(18)
sec.different_first_page_header_footer = True
hp = sec.header.paragraphs[0]
hp.text = "BCSE306L  |  Course Project Final Report"
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for r in hp.runs:
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
run.font.size = Pt(8.5)
for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    run._r.append(el)

# tables: thin grey borders, tinted header row
text_width = sec.page_width - sec.left_margin - sec.right_margin
for t in doc.tables:
    tblPr = t._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "C3CFE0")):
            e.set(qn(k), v)
        borders.append(e)
    tblPr.append(borders)
    w = tblPr.find(qn("w:tblW"))
    if w is None:
        w = OxmlElement("w:tblW")
        tblPr.append(w)
    w.set(qn("w:type"), "pct")
    w.set(qn("w:w"), "5000")
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", 40), ("left", 90), ("bottom", 40), ("right", 90)):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(val))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    first = t.rows[0]
    if first._tr.trPr is not None and first._tr.trPr.find(qn("w:tblHeader")) is not None:
        for cell in first.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", "EAF0F8")):
                shd.set(qn(k), v)
            tcPr.append(shd)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.bold = True
                    r.font.color.rgb = NAVY

# images: fit the text width, centre, keep the caption with the figure
for shape in doc.inline_shapes:
    if shape.width > text_width:
        ratio = text_width / shape.width
        shape.height = Emu(int(shape.height * ratio))
        shape.width = Emu(int(text_width))
for p in doc.paragraphs:
    if p._p.xpath(".//w:drawing"):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True

doc.core_properties.title = "VoiceChat: BCSE306L Course Project Final Report"
doc.core_properties.author = "Project team"
doc.save(OUT)
RAW.unlink()
REF.unlink()
LUA.unlink()
print("wrote", OUT)
