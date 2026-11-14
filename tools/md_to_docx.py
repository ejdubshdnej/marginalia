"""把 docs/REPORT.md 转成 Word 版本。

只处理本报告实际用到的 Markdown 子集：ATX 标题、段落、无序/有序列表、
管道表格、围栏代码块、行间图片、水平线；行内的 **粗体**、`代码`、*斜体*。

用法：
    python tools/md_to_docx.py \
        --src docs/REPORT.md \
        --dst docs/REPORT.docx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---------------------------------------------------------------- 版式常量
BODY_FONT = "Times New Roman"
BODY_FONT_CJK = "宋体"
HEAD_FONT_CJK = "黑体"
CODE_FONT = "Consolas"

BODY_SIZE = Pt(10.5)
CODE_SIZE = Pt(8.5)
TABLE_SIZE = Pt(9)

HEAD_SPEC = {
    1: (Pt(18), True, WD_ALIGN_PARAGRAPH.CENTER, Pt(0), Pt(12)),
    2: (Pt(14), True, WD_ALIGN_PARAGRAPH.LEFT, Pt(14), Pt(8)),
    3: (Pt(12), True, WD_ALIGN_PARAGRAPH.LEFT, Pt(10), Pt(6)),
    4: (Pt(10.5), True, WD_ALIGN_PARAGRAPH.LEFT, Pt(8), Pt(4)),
}

CENTER_PREFIX = ("摘　要", "参考文献", "附录")


def set_run_font(run, size=BODY_SIZE, bold=False, cjk=BODY_FONT_CJK,
                 latin=BODY_FONT, color=None):
    run.font.name = latin
    run.font.size = size
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cjk)
    if color is not None:
        run.font.color.rgb = color


def set_para_spacing(p, before=Pt(0), after=Pt(4), line=1.45):
    pf = p.paragraph_format
    pf.space_before = before
    pf.space_after = after
    pf.line_spacing = line


# ---------------------------------------------------------------- 行内解析
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")


def add_inline(paragraph, text, size=BODY_SIZE, base_bold=False):
    """把一段含 **粗体** / `代码` / *斜体* 的文本写进 paragraph。

    粗体内部可以再嵌行内代码（如 **必须能在 `chunks.jsonl` 中查到**），
    因此粗体分支递归调用本函数。
    """
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            inner = piece[2:-2]
            if "`" in inner:
                add_inline(paragraph, inner, size=size, base_bold=True)
            else:
                r = paragraph.add_run(inner)
                set_run_font(r, size, bold=True)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = paragraph.add_run(piece[1:-1])
            set_run_font(r, Pt(size.pt - 1.0), bold=base_bold, latin=CODE_FONT)
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = paragraph.add_run(piece[1:-1])
            set_run_font(r, size, bold=base_bold)
            r.italic = True
        else:
            r = paragraph.add_run(piece)
            set_run_font(r, size, bold=base_bold)


def split_cells(line: str) -> list[str]:
    """按未转义的竖线切分表格行，去掉首尾空壳。"""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip().replace("\\|", "|") for c in line.split("|")]


def is_separator(line: str) -> bool:
    cells = split_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


# ---------------------------------------------------------------- 块级渲染
def add_table(doc, rows: list[list[str]]):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    table = doc.add_table(rows=1, cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # 跨页时重复表头，并让表格占满正文宽度
    tblPr = table._tbl.tblPr
    tblW = tblPr.makeelement(qn("w:tblW"), {qn("w:type"): "pct", qn("w:w"): "5000"})
    tblPr.append(tblW)
    layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "autofit"})
    tblPr.append(layout)

    hdr_row = table.rows[0]
    trPr = hdr_row._tr.get_or_add_trPr()
    trPr.append(trPr.makeelement(qn("w:tblHeader"), {qn("w:val"): "true"}))

    for j, text in enumerate(header):
        cell = table.rows[0].cells[j]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, Pt(2), Pt(2), line=1.1)
        add_inline(p, text, size=TABLE_SIZE, base_bold=True)

    for row in body:
        cells = table.add_row().cells
        for j in range(ncol):
            text = row[j] if j < len(row) else ""
            p = cells[j].paragraphs[0]
            # 纯数字或短文本居中，长文本左对齐
            if len(re.findall(r"[\u4e00-\u9fff]", text)) <= 6 and len(text) <= 14:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_para_spacing(p, Pt(1), Pt(1), line=1.1)
            add_inline(p, text, size=TABLE_SIZE)

    doc.add_paragraph()  # 表后留白


def add_code_block(doc, lines: list[str]):
    for line in lines:
        p = doc.add_paragraph()
        set_para_spacing(p, Pt(0), Pt(0), line=1.0)
        p.paragraph_format.left_indent = Cm(0.6)
        r = p.add_run(line if line else " ")
        set_run_font(r, CODE_SIZE, latin=CODE_FONT)
        shading = p._p.get_or_add_pPr()
        shd = shading.makeelement(
            qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): "F4F5F7"})
        shading.append(shd)
    doc.add_paragraph()


GREEK = {
    r"\in": "∈", r"\sum": "Σ", r"\prod": "∏", r"\cdot": "·",
    r"\times": "×", r"\le": "≤", r"\ge": "≥", r"\approx": "≈",
    r"\alpha": "α", r"\beta": "β", r"\tau": "τ", r"\theta": "θ",
    r"\epsilon": "ε", r"\sigma": "σ", r"\Sigma": "Σ",
}


def _read_group(s: str, i: int) -> tuple[str, int]:
    """从 s[i] 读取一个 {…} 组或单个字符，返回 (内容, 新位置)。"""
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    return (s[i] if i < len(s) else ""), i + 1


def latex_to_runs(latex: str) -> list[tuple[str, str]]:
    """把受限子集的 LaTeX 展开成 [(文本, 角色)] 序列。

    角色取值：plain（正体）、var（斜体变量）、sub（下标）、sup（上标）。
    只支持 \\text \\sum \\frac \\_ ^ 与希腊字母，够本文的公式用。
    """
    out: list[tuple[str, str]] = []
    i, n = 0, len(latex)

    def push(items, role="plain"):
        for text, r in items:
            out.append((text, role if r == "plain" else r))

    while i < n:
        ch = latex[i]
        if ch == "\\":
            j = i + 1
            while j < n and latex[j].isalpha():
                j += 1
            cmd = latex[i:j]
            if cmd == r"\text":
                body, i = _read_group(latex, j)
                out.append((body, "plain"))
            elif cmd == r"\frac":
                num, i = _read_group(latex, j)
                den, i = _read_group(latex, i)
                simple = re.fullmatch(r"\s*\d+\s*", num) is not None
                if not simple:
                    out.append(("(", "plain"))
                push(latex_to_runs(num))
                if not simple:
                    out.append((")", "plain"))
                out.append((" / (", "plain"))
                push(latex_to_runs(den))
                out.append((")", "plain"))
            elif cmd in GREEK:
                out.append((GREEK[cmd], "var"))
                i = j
            elif cmd == r"\sum":
                out.append(("Σ", "plain"))
                i = j
            else:  # 未知命令：只保留字符本体
                out.append((cmd.lstrip("\\"), "plain"))
                i = j
        elif ch == "_":
            body, i = _read_group(latex, i + 1)
            push(latex_to_runs(body), "sub")
        elif ch == "^":
            body, i = _read_group(latex, i + 1)
            push(latex_to_runs(body), "sup")
        elif ch == " ":
            out.append((" ", "plain"))
            i += 1
        elif ch.isalpha():
            j = i
            while j < n and latex[j].isalpha():
                j += 1
            out.append((latex[i:j], "var"))
            i = j
        else:
            out.append((ch, "plain"))
            i += 1
    return out


def add_equation(doc, latex: str):
    """居中显示行间公式。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p, Pt(8), Pt(8), line=1.3)
    for text, role in latex_to_runs(latex):
        r = p.add_run(text)
        set_run_font(r, Pt(12), latin="Cambria Math", cjk=BODY_FONT_CJK)
        if role == "var":
            r.italic = True
        elif role == "sub":
            r.font.subscript = True
            r.italic = True
        elif role == "sup":
            r.font.superscript = True


def add_image(doc, path: Path, caption: str | None):
    if not path.is_file():
        doc.add_paragraph(f"[缺图：{path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p, Pt(8), Pt(2))
    # A4 21 cm 减去左右各 3 cm 页边距，正文宽度 15 cm
    p.add_run().add_picture(str(path), width=Cm(14.9))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(cap, Pt(0), Pt(10))
        r = cap.add_run(caption)
        set_run_font(r, Pt(9), latin=BODY_FONT)


def add_heading(doc, level: int, text: str):
    size, bold, align, before, after = HEAD_SPEC[level]
    # 副标题与"摘要/参考文献/附录"这类一级块居中
    if text.startswith("——") or text.startswith(CENTER_PREFIX):
        align = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(style=f"Heading {level}" if level < 4 else "Heading 4")
    p.alignment = align
    set_para_spacing(p, before, after, line=1.3)
    r = p.add_run(text)
    set_run_font(r, size, bold=bold, cjk=HEAD_FONT_CJK,
                 color=RGBColor(0x1F, 0x2A, 0x37))
    # Heading 样式自带蓝色，强制改黑
    r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def add_page_number_footer(doc):
    """页脚居中页码，形如 `第 3 页 / 共 12 页`。

    Word 的域必须拆成 begin / instrText / separate / 结果 / end 五个独立 run，
    全部塞进同一个 run 里时部分阅读器不认。
    """
    for section in doc.sections:
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(p, Pt(0), Pt(0), line=1.0)

        def field(instr):
            for kind in ("begin", "instr", "separate", "text", "end"):
                r = p.add_run()
                set_run_font(r, Pt(9))
                if kind == "begin":
                    r._r.append(r._r.makeelement(
                        qn("w:fldChar"), {qn("w:fldCharType"): "begin"}))
                elif kind == "instr":
                    el = r._r.makeelement(qn("w:instrText"),
                                          {qn("xml:space"): "preserve"})
                    el.text = f" {instr} "
                    r._r.append(el)
                elif kind == "separate":
                    r._r.append(r._r.makeelement(
                        qn("w:fldChar"), {qn("w:fldCharType"): "separate"}))
                elif kind == "text":
                    r.text = "1"
                else:
                    r._r.append(r._r.makeelement(
                        qn("w:fldChar"), {qn("w:fldCharType"): "end"}))

        for label, code in (("第 ", "PAGE"), (" 页 / 共 ", "NUMPAGES")):
            r = p.add_run(label)
            set_run_font(r, Pt(9))
            field(code)
        r = p.add_run(" 页")
        set_run_font(r, Pt(9))


def add_toc(doc):
    """插入可更新的目录域（Word / WPS 中右键"更新域"即可填充页码）。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(p, Pt(6), Pt(10), line=1.3)
    r = p.add_run("目　录")
    set_run_font(r, Pt(14), bold=True, cjk=HEAD_FONT_CJK)

    holder = doc.add_paragraph()
    set_para_spacing(holder, Pt(0), Pt(6), line=1.5)

    def fld(kind, text=None, instr=None):
        run = holder.add_run()
        set_run_font(run, Pt(10))
        if kind == "instr":
            el = run._r.makeelement(qn("w:instrText"),
                                    {qn("xml:space"): "preserve"})
            el.text = f" {instr} "
            run._r.append(el)
        elif kind == "text":
            run.text = text
        else:
            run._r.append(run._r.makeelement(
                qn("w:fldChar"), {qn("w:fldCharType"): kind}))
        return run

    fld("begin")
    fld("instr", instr=r'TOC \o "1-3" \h \z \u')
    fld("separate")
    hint = fld("text",
               text="（目录域：在 Word / WPS 中按 Ctrl+A 后 F9，或右键此处选择"
                    "“更新域”，即可按一至三级标题生成带页码的目录。）")
    hint.font.color.rgb = RGBColor(0x88, 0x8C, 0x94)
    fld("end")

    doc.add_page_break()


def add_rule(doc):
    p = doc.add_paragraph()
    set_para_spacing(p, Pt(2), Pt(2), line=1.0)
    pPr = p._p.get_or_add_pPr()
    borders = pPr.makeelement(qn("w:pBdr"), {})
    bottom = borders.makeelement(qn("w:bottom"), {
        qn("w:val"): "single", qn("w:sz"): "6",
        qn("w:space"): "1", qn("w:color"): "C8CDD4"})
    borders.append(bottom)
    pPr.append(borders)


def add_list_item(doc, text: str, ordered: bool, indent: int = 0, num: int = 0):
    p = doc.add_paragraph()
    set_para_spacing(p, Pt(0), Pt(3), line=1.4)
    p.paragraph_format.left_indent = Cm(0.75 * (indent + 1))
    p.paragraph_format.first_line_indent = Cm(-0.55 if ordered else -0.4)
    marker = f"{num}. " if ordered else "· "
    r = p.add_run(marker)
    set_run_font(r, BODY_SIZE)
    add_inline(p, text)


def add_body(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_para_spacing(p, Pt(0), Pt(6), line=1.5)
    p.paragraph_format.first_line_indent = Cm(0.74)
    add_inline(p, text)


# ---------------------------------------------------------------- 主流程
META_PREFIX = ("项目类型", "语料规模", "技术栈", "文档版本")


def render_front_matter(doc, lines: list[str]):
    """封面区块：# 标题 / ## 副标题 / 英文题名 / 元信息四行。"""
    for line in lines:
        text = line.strip()
        if not text:
            continue
        m = re.match(r"^(#{1,2})\s+(.*)$", text)
        if m:
            add_heading(doc, len(m.group(1)), m.group(2).strip())
            continue
        if text.startswith("**") and text.endswith("**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_para_spacing(p, Pt(10), Pt(10), line=1.3)
            r = p.add_run(text[2:-2])
            set_run_font(r, Pt(11), bold=True)
            continue
        if text.startswith(META_PREFIX):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_para_spacing(p, Pt(2), Pt(2), line=1.35)
            add_inline(p, text, size=Pt(9.5))
            continue
        add_body(doc, text)


def convert(src: Path, dst: Path):
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CJK)

    for section in doc.sections:
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(3.0)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)

    lines = src.read_text(encoding="utf-8").splitlines()
    base_dir = src.parent

    # 首个水平线之前是封面区（标题、副标题、英文题名、元信息）
    front_end = 0
    for idx, ln in enumerate(lines):
        if re.fullmatch(r"-{3,}", ln.strip()):
            front_end = idx
            break
    render_front_matter(doc, lines[:front_end])
    add_toc(doc)

    i = front_end
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        # 空行
        if not line.strip():
            i += 1
            continue

        # 水平线
        if re.fullmatch(r"-{3,}", line.strip()):
            add_rule(doc)
            i += 1
            continue

        # 围栏代码
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code_block(doc, buf)
            continue

        # 行间公式 $$...$$
        if line.strip().startswith("$$"):
            add_equation(doc, line.strip().strip("$").strip())
            i += 1
            continue

        # 图片
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if m:
            alt, rel = m.group(1), m.group(2)
            img = (base_dir / rel).resolve()
            caption = alt.replace("　", " ") if alt else None
            add_image(doc, img, caption)
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            add_heading(doc, level, m.group(2).strip())
            i += 1
            continue

        # 表格
        if line.lstrip().startswith("|") and i + 1 < n and is_separator(lines[i + 1]):
            rows = [split_cells(line)]
            i += 2
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(split_cells(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # 列表
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if m:
            indent = len(m.group(1)) // 2
            tok = m.group(2)
            add_list_item(doc, m.group(3), ordered=tok[0].isdigit(),
                          indent=indent,
                          num=int(tok[:-1]) if tok[0].isdigit() else 0)
            i += 1
            continue

        # 普通段落（连续非空行合并）
        buf = [line.strip()]
        i += 1
        while i < n:
            nxt = lines[i].rstrip()
            if (not nxt.strip() or nxt.lstrip().startswith(("|", "#", "```", "!["))
                    or re.fullmatch(r"-{3,}", nxt.strip())
                    or re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[i])):
                break
            buf.append(nxt.strip())
            i += 1
        add_body(doc, "".join(buf))

    dst.parent.mkdir(parents=True, exist_ok=True)
    add_page_number_footer(doc)
    doc.save(str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/REPORT.md")
    ap.add_argument("--dst", default="docs/REPORT.docx")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    src = (root / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    dst = (root / args.dst) if not Path(args.dst).is_absolute() else Path(args.dst)
    out = convert(src, dst)
    print(f"written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
