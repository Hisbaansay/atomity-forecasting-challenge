#!/usr/bin/env python3
"""Render the report's Markdown source with six core pages and a figure appendix.

Run: .venv/bin/python scripts/render_report.py
Requires reportlab and pypdf. No report prose is maintained in this exporter.
"""

from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from atomity_forecasting.typography import times_new_roman_files


ROOT = Path(__file__).resolve().parents[1]
INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
PICTURE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def report_fonts() -> tuple[str, str]:
    """Embed installed Times New Roman; use PDF Times only when unavailable."""
    paths = times_new_roman_files()
    if paths:
        names = {"regular": "ReportTimes", "bold": "ReportTimes-Bold",
                 "italic": "ReportTimes-Italic", "bold_italic": "ReportTimes-BoldItalic"}
        for style, name in names.items():
            pdfmetrics.registerFont(TTFont(name, str(paths[style])))
        pdfmetrics.registerFontFamily(names["regular"], normal=names["regular"], bold=names["bold"],
                                      italic=names["italic"], boldItalic=names["bold_italic"])
        return names["regular"], names["bold"]
    return "Times-Roman", "Times-Bold"


def render(source: Path, output: Path) -> tuple[int, int]:
    regular_font, bold_font = report_fonts()
    text = source.read_text(encoding="utf-8")
    figures: dict[str, str] = {}
    main_figures = {match[2] for match in PICTURE.finditer(text)}
    for match in PICTURE.finditer(text):
        figures.setdefault(match[2], match[1])
    for match in LINK.finditer(text):
        if Path(match[2]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            figures.setdefault(match[2], match[1])
    anchors = {name: f"figure-{i}" for i, name in enumerate(figures, 1)}

    def inline(value: str) -> str:
        parts = INLINE.split(value)
        for i in range(len(parts)):
            part = parts[i]
            if part.startswith("`") and part.endswith("`"):
                parts[i] = escape(part[1:-1])
            elif part.startswith("**") and part.endswith("**"):
                parts[i] = "<b>" + escape(part[2:-2]) + "</b>"
            elif (match := LINK.fullmatch(part)) is not None:
                label, target = match.groups()
                if target in anchors:
                    target = "#" + anchors[target]
                parts[i] = f'<link href="{escape(target, quote=True)}" color="#24556b">{inline(label)}</link>'
            else:
                parts[i] = escape(part)
        return "".join(parts)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportBody", fontName=regular_font, fontSize=11, leading=14.2,
        spaceAfter=7, alignment=TA_JUSTIFY, textColor=colors.HexColor("#202b33"),
    ))
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["ReportBody"], fontSize=20,
        leading=24, spaceAfter=16, fontName=bold_font, alignment=TA_LEFT,
        textColor=colors.HexColor("#173c50"),
    ))
    styles.add(ParagraphStyle(
        "ReportHeading", parent=styles["ReportBody"], fontName=bold_font, alignment=TA_LEFT,
        fontSize=13, leading=16, spaceBefore=9, spaceAfter=9, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "ReportCaption", parent=styles["ReportBody"], fontSize=9.5,
        leading=12, spaceAfter=9, alignment=TA_LEFT, textColor=colors.HexColor("#455d6b"),
    ))
    styles.add(ParagraphStyle(
        "ReportCell", parent=styles["ReportBody"], fontSize=9.5,
        leading=11.5, spaceAfter=0, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "ReportBullet", parent=styles["ReportBody"], leftIndent=12,
        firstLineIndent=-9,
    ))
    width, height = 8.5 * inch, 11 * inch
    margin = 0.6 * inch
    usable_width = width - 2 * margin

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cad4da"))
        canvas.setLineWidth(0.4)
        canvas.line(margin, height - 29, width - margin, height - 29)
        canvas.setFillColor(colors.HexColor("#536774"))
        canvas.setFont(regular_font, 8)
        canvas.drawString(margin, height - 22, "CLOUD WORKLOAD FORECASTING")
        canvas.drawString(margin, 24, "Training-only validation | Azure data-v1")
        canvas.drawRightString(width - margin, 24, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(output), pagesize=(width, height), leftMargin=margin,
        rightMargin=margin, topMargin=margin, bottomMargin=margin,
        title="Cloud Workload Forecasting Report", author="",
        invariant=1, pageCompression=1,
    )
    doc.addPageTemplates(PageTemplate(
        id="report", frames=[Frame(
            margin, margin, usable_width, height - 2 * margin,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )], onPage=furniture,
    ))
    story = []
    lines = text.splitlines()
    i = 0
    appendix_seen = False
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if (heading := re.match(r"^(#{1,6})\s+(.+)$", line)) is not None:
            level, title = heading.groups()
            if title.startswith("Appendix:"):
                story.append(PageBreak())
                appendix_seen = True
            elif len(level) == 2 and re.match(r"^(3|4|6|7|8)\. ", title):
                # Explicit section groups keep the core readable and predictable.
                story.append(PageBreak())
            style = "ReportTitle" if len(level) == 1 else "ReportHeading"
            story.append(Paragraph(inline(title), styles[style]))
        elif (picture := PICTURE.fullmatch(line)) is not None:
            caption, target = picture.groups()
            image = Image(str(source.parent / target))
            max_height = 3.5 if Path(target).stem == "backtest_cpu" else 2.5
            scale = min(usable_width / image.imageWidth, max_height * inch / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            label = Paragraph(
                f'<a name="{anchors[target]}"/>' + inline(caption), styles["ReportCaption"],
            )
            story.append(KeepTogether([image, Spacer(1, 5), label]))
        elif line.startswith("|"):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            data = []
            for row in rows:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                data.append([Paragraph(inline(cell), styles["ReportCell"]) for cell in cells])
            columns = len(data[0])
            # Text-heavy model labels need more room than numeric diagnostics.
            first_share = 0.30 if columns <= 4 else 0.13
            col_widths = [usable_width * first_share] + [
                usable_width * (1 - first_share) / (columns - 1)
            ] * (columns - 1) if columns > 1 else [usable_width]
            table = Table(data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#9aadb9")),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + inline(line[2:]), styles["ReportBullet"]))
        else:
            paragraph = [line]
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |!\[)", lines[i]):
                paragraph.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), styles["ReportBody"]))

    if not appendix_seen:
        raise ValueError("Expected an explicit Appendix heading in the source report")
    appendix_figures = [(target, caption) for target, caption in figures.items() if target not in main_figures]
    for index, (target, caption) in enumerate(appendix_figures):
        if index % 2 == 0:
            story.extend([PageBreak(), Paragraph("Appendix: Figures", styles["ReportHeading"])])
        image = Image(str(source.parent / target))
        scale = min(usable_width / image.imageWidth, 3.7 * inch / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        label = Paragraph(
            f'<a name="{anchors[target]}"/>' + inline(caption), styles["ReportCaption"],
        )
        story.append(KeepTogether([label, image, Spacer(1, 14)]))
    doc.build(story)
    pages = PdfReader(output).pages
    appendix_page = next(
        i for i, page in enumerate(pages)
        if "Appendix: Supporting Evidence" in page.extract_text()
    )
    if appendix_page != 6:
        raise ValueError(f"Expected 6 core pages, found {appendix_page}; review section layout")
    return appendix_page, len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final_report.pdf")
    args = parser.parse_args()
    core, total = render(args.source.resolve(), args.output.resolve())
    print(f"Rendered {args.output}: {core} core pages + {total - core} appendix pages = {total} total pages")


if __name__ == "__main__":
    main()
