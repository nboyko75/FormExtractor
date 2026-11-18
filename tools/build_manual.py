"""Build the bundled offline HTML/PDF manual from one content source."""
# ruff: noqa: E501

import shutil
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/formextractor/manual"
PAGES = [
    (
        "Getting started",
        [
            (
                "p",
                "FormExtractor reads selected fields from scanned forms and places the results in a table. Define areas on one source document, then recognize the matching files in the same folder. Each file becomes a row; each area becomes a column.",
            ),
            ("image", "Main_01.png"),
            (
                "note",
                "Example recognition session using the supplied screenshot. Results shown are examples, not verified transcriptions. The screenshot shows the older About menu; the current menu is Help > App manual / About. Click the image in the HTML manual to view it at full size.",
            ),
            ("h", "Your first batch"),
            (
                "p",
                "1. Import a representative form with the folder button or File > Import (Ctrl+O).",
            ),
            ("p", "2. Choose the recognition language and a file mask, such as *.png."),
            ("p", "3. Click + beside Text, Number or Date and drag a rectangle around each value."),
            ("p", "4. Select Save areas if you want images of the matched fields for checking."),
            ("p", "5. Click Recognize. Review the values and any errors as the table fills."),
            ("p", "6. Correct the selected output tab if necessary, then choose Save (Ctrl+S)."),
        ],
    ),
    (
        "Preview and recognition areas",
        [
            ("h", "Find your way around"),
            (
                "p",
                "The left pane shows the source document. Its full path appears beside Import. The right pane contains Text and JSON results. Drag the raised divider between the panes to change their widths. The bottom status line shows progress and the current file.",
            ),
            (
                "p",
                "Use the magnifying-glass buttons or mouse wheel to zoom from 25% to 400% of the fitted size. Scroll or drag to inspect a large page. Previous and Next browse PDF pages and multi-frame images. Area coordinates remain correct when you zoom.",
            ),
            ("h", "Add a field"),
            (
                "p",
                "Choose the + button in the appropriate type group. The pointer becomes a crosshair. Drag around the value, including the full height of its characters. Keep unrelated labels, neighbouring fields and extra rows outside the rectangle. Select one value per Number or Date area.",
            ),
            (
                "p",
                "Text areas are red, Number areas are blue, and Date areas are green. Every rectangle has a number and type label. Numbers determine result-column order, not the rectangle's position on the page.",
            ),
            ("h", "Remove or redraw a field"),
            (
                "p",
                "Choose the minus button for the area's type, then click the rectangle to remove it. Removal affects that type on the current page only. To correct a rectangle, remove it and draw a new one. Remaining area numbers are not renumbered. Press Escape or click the active tool again to return to panning.",
            ),
            ("h", "Choose a useful template"),
            (
                "p",
                "Use a clear, complete example of the form layout. Matching uses the surrounding form structure to locate the fields in other files, allowing movement, rotation, scale changes and photographed-page perspective. The other files should use the same form layout and corresponding page numbers.",
            ),
            (
                "p",
                "A small field can still match when the rest of the page has enough distinctive structure. A blank page, heavily cropped form or a different layout may not align reliably. A failed match is reported for that area; other files can continue.",
            ),
            ("h", "Remembered areas"),
            (
                "p",
                "Areas are saved automatically for each source file's full path. Importing that file again restores its areas. On startup, the last available document and page reopen with their saved selections. Renaming or moving a source changes its path, so its old selections may not be restored automatically.",
            ),
        ],
    ),
    (
        "Text, numbers and dates",
        [
            ("h", "Text: words and identifiers"),
            (
                "p",
                "Use Text for names, vehicle plates, addresses and identifiers. Text preserves letters and meaningful leading zeros. For a postal code such as 00123, choose Text rather than Number. Formatting tags and leading heading markers are removed from recognized values.",
            ),
            ("h", "Number: a numeric value"),
            (
                "p",
                "Number accepts digits, decimal dots or commas, signs and spaces. The result is normalized: -001,50 becomes -1.5; +12.00 becomes 12; 319 320 becomes 319320. Units and surrounding text are removed when one clear number is present. Multiple or ambiguous expressions such as 12/34 are reported as errors.",
            ),
            ("h", "Date: a single calendar date"),
            (
                "p",
                "Dates are always displayed as DD.MM.YYYY. Extra punctuation and printed year suffixes such as р. are removed. Invalid dates are reported rather than silently changed to a different calendar date.",
            ),
            (
                "p",
                "Accepted examples: 21.03.2026, 21/03/2026, 21-03-2026, 21 03 2026 and 2026-03-21 all become 21.03.2026. Unambiguous month-first dates such as 03/21/2026 are also accepted.",
            ),
            (
                "p",
                "Compact DDMMYYYY, YYYYMMDD and DDMMYY forms are supported when their interpretation is clear. Recognized English, Ukrainian and Russian month names and common abbreviations are accepted, for example 21 March 2026 or 21 березня 2026.",
            ),
            (
                "note",
                "Ambiguous dates use day-first order: 06/04/2026 means 6 April. Two-digit years 00-49 mean 2000-2049; 50-99 mean 1950-1999. Use a four-digit year when a different century is intended.",
            ),
            ("h", "Check the recognition"),
            (
                "p",
                "Select the language used by the document in the Recognition language box below the preview. Handwriting, blur, shadows, faint characters and table lines can affect accuracy. Check important dates and numbers against the original even when no error is reported.",
            ),
            (
                "p",
                "If a date fails, include the complete date in the rectangle and exclude neighbouring text or table rows. The app retries empty or invalid date readings to handle suffixes that can be mistaken for digits. It cannot reconstruct characters missing from the image.",
            ),
        ],
    ),
    (
        "Batch recognition and saving",
        [
            ("h", "Choose the files"),
            (
                "p",
                "File mask selects files in the imported document's folder. Use * for all supported files, *.pdf for PDFs, or *.png;*.jpg for multiple patterns. A question mark matches one character: ????.jpg matches four-character names. Matching is case-insensitive, alphabetical and does not search subfolders.",
            ),
            (
                "p",
                "Supported inputs are PDF, JPG/JPEG, PNG, TIFF, BMP and WebP. Multi-page PDFs and multi-frame images can have areas on different pages. A missing corresponding page is reported for that file. Unlock password-protected PDFs before importing them.",
            ),
            ("h", "Run, monitor and stop"),
            (
                "p",
                "Recognize becomes available when a document has at least one area. The first run may download OCR models; allow time for it to finish. Results appear after each area. Controls that would change an active job are temporarily disabled. Stop ends the worker and keeps results already displayed; you can save them or start recognition again.",
            ),
            ("h", "Save areas for checking"),
            (
                "p",
                "Enable Save areas before recognition to save a PNG of the matched crops in an automatically created areas subfolder beside each input, such as areas/invoice_areas.png. Crops are stacked by area number. An unmatched crop is omitted; a matched crop can still be saved if OCR fails. These generated images are excluded from future batches. A later run replaces the corresponding areas image.",
            ),
            ("h", "Review the table"),
            (
                "p",
                "The Text tab shows an aligned table. JSON contains the source, selected areas and file rows with their values and optional errors. Columns follow ascending area numbers. A missing or failed value can be null in JSON. Successful values, including normalized numbers and dates, are stored as strings.",
            ),
            (
                "p",
                "Both editors are editable, but their edits are independent. Changing Text does not update JSON, and changing JSON does not update Text. Review the exact tab you intend to save. Partial results can contain unfinished or failed fields.",
            ),
            ("h", "Save the result"),
            (
                "p",
                "Choose Save or Ctrl+S. The active tab supplies the default format. The file-type choice determines whether the Text editor is saved as .txt or the JSON editor as .json. JSON must be valid before it can be saved. Save to a new result file; the input document is protected from being overwritten.",
            ),
        ],
    ),
    (
        "Menus, settings and troubleshooting",
        [
            ("h", "Menus and keyboard shortcuts"),
            (
                "p",
                "File > Import (Ctrl+O) opens a source. File > Recognize starts the batch. File > Save (Ctrl+S) saves results. File > Settings changes the interface language. File > Exit closes the app and stops active recognition.",
            ),
            (
                "p",
                "Help > App manual (F1) opens this offline English manual in your browser. Use the PDF link to open or save the printable guide. Help > About shows the application version and author contact details. Escape exits an active area tool or closes a settings/about dialog.",
            ),
            ("h", "Language and preferences"),
            (
                "p",
                "File > Settings > Main offers English, German, French, Spanish, Portuguese, Italian, Ukrainian and Russian interface languages. A changed interface language restarts the desktop session after confirmation of unsaved edits. Cancel discards changes. The interface language is separate from Recognition language.",
            ),
            (
                "p",
                "The app remembers window size and position, pane widths, zoom, OCR language, file mask, Save areas, and the last source/page. Area selections are saved per source path. Recognition results are not a substitute for a saved output file; save any work you need before closing.",
            ),
            ("h", "Common problems"),
            (
                "p",
                "Recognize is disabled: import a document and add at least one area. No matching files: check the mask and source folder. Model download is slow or fails: check the connection, wait for the first download, then retry. A newly selected OCR language may need additional models.",
            ),
            (
                "p",
                "No reliable match or wrong crop: enable Save areas, compare the crops with the source, and confirm that the files share the template layout and page order. Try a clearer, more complete template. If the crop is correct but its value is wrong, review the area type, language and image quality.",
            ),
            (
                "p",
                "Date/number error: select a single complete value and exclude adjacent fields. Do not remove leading zeros with Number when they identify an item; use Text instead. Save error: check the destination permissions, extension and JSON syntax. Manual will not open: use the file path shown in the error message or reinstall a complete app build.",
            ),
            ("h", "Local processing and support"),
            (
                "p",
                "Recognition runs locally. OCR model files are downloaded on first use; once the required models are cached, recognition can work offline. Scans are not submitted to a remote recognition service. This manual and its screenshot also work offline.",
            ),
            (
                "p",
                "For support, use the contact details in Help > About. Include the app version, a description of the issue and the error text. Share source documents or saved crops only when you intend to share their contents.",
            ),
        ],
    ),
]


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    screenshot = ROOT / "store/Screenshots/Main_01.png"
    if screenshot.is_file():
        shutil.copyfile(screenshot, DEST / "Main_01.png")
    elif not (DEST / "Main_01.png").is_file():
        raise FileNotFoundError(screenshot)
    nav = "".join(
        f'<a href="#chapter-{i}">{i}. {escape(title)}</a>' for i, (title, _) in enumerate(PAGES, 1)
    )
    body = []
    for i, (title, blocks) in enumerate(PAGES, 1):
        body.append(f'<section id="chapter-{i}"><h2>{i}. {escape(title)}</h2>')
        for kind, value in blocks:
            if kind == "image":
                body.append(
                    '<a href="Main_01.png"><img src="Main_01.png" alt="FormExtractor showing selected Text, Number and Date areas beside a batch results table" width="1557" height="979"></a>'
                )
            else:
                tag = "h3" if kind == "h" else "p"
                body.append(f'<{tag} class="{kind}">{escape(value)}</{tag}>')
        body.append("</section>")
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;color:#1d2c3d;font:17px/1.65 'Segoe UI',Arial,sans-serif}nav{position:fixed;width:265px;height:100vh;padding:30px 24px;background:#132b45;color:white;overflow:auto}nav strong{font-size:23px}nav a{display:block;color:#d7e6f8;margin:18px 0;line-height:1.4}a{color:#0759a5;text-underline-offset:3px}main{max-width:1300px;margin-left:265px;padding:40px 48px}h1{font-size:40px;line-height:1.15;margin:8px 0 18px}h2{font-size:27px;line-height:1.3;color:#183d61}h3{font-size:20px;margin:28px 0 8px}section{background:white;border:1px solid #dce3eb;border-radius:10px;padding:24px 32px;margin:28px 0;scroll-margin:20px}p{margin:12px 0}img{width:100%;height:auto;border:1px solid #c6d0dc;border-radius:5px}.note{font-size:15px;background:#edf4fb;border-left:4px solid #2076b4;padding:12px 16px}.pdf{display:inline-block;padding:9px 18px;border-radius:5px;background:#0759a5;color:white;text-decoration:none}a:focus-visible{outline:3px solid #e9a423;outline-offset:4px}.skip{position:absolute;left:-10000px}.skip:focus{left:280px;top:0;background:white;z-index:2}@media(max-width:850px){nav{position:static;width:auto;height:auto}nav a{margin:10px 0}main{margin:0;padding:24px 16px}section{padding:20px}h1{font-size:32px}}@media print{nav,.pdf,.skip{display:none}main{margin:0;padding:0}section{border:0;break-before:page}body{background:white;font-size:11pt}img{max-height:110mm;object-fit:contain}}"""
    html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FormExtractor - App manual</title><style>{css}</style></head><body><a class="skip" href="#content">Skip to manual</a><nav aria-label="Contents"><strong>FormExtractor</strong><p>App manual · English</p>{nav}</nav><main id="content"><h1>FormExtractor<br>App manual</h1><p>Select fields. Recognize a folder. Review and save.</p><a class="pdf" href="FormExtractor_User_Manual.pdf">Open printable PDF</a>{"".join(body)}<footer>FormExtractor · Offline user manual</footer></main></body></html>'
    (DEST / "index.html").write_text(html, encoding="utf-8")
    fonts = Path("C:/Windows/Fonts")
    if fonts.exists():
        regular, bold = fonts / "arial.ttf", fonts / "arialbd.ttf"
    else:
        fonts = Path("/usr/share/fonts/truetype/dejavu")
        regular, bold = fonts / "DejaVuSans.ttf", fonts / "DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("Manual", str(regular)))
    pdfmetrics.registerFont(TTFont("ManualBold", str(bold)))
    styles = {
        k: ParagraphStyle(
            k,
            fontName="ManualBold" if k in ("title", "h") else "Manual",
            fontSize=size,
            leading=leading,
            spaceAfter=8,
            textColor=colors.HexColor("#1d2c3d"),
        )
        for k, size, leading in (("title", 22, 28), ("h", 13, 17), ("p", 10, 14), ("note", 8, 11))
    }
    styles["h"].spaceBefore = 10
    styles["h"].keepWithNext = True
    story = []
    for i, (title, blocks) in enumerate(PAGES, 1):
        if i > 1:
            story.append(PageBreak())
        story.append(Paragraph(f"{i}. {escape(title)}", styles["title"]))
        for kind, value in blocks:
            if kind == "image":
                story.extend(
                    [Image(str(DEST / value), width=495, height=495 * 979 / 1557), Spacer(1, 8)]
                )
            else:
                story.append(Paragraph(escape(value), styles[kind]))

    def footer(canvas, doc):
        canvas.setFont("Manual", 9)
        canvas.setFillColor(colors.HexColor("#476078"))
        canvas.drawString(50, 812, "FormExtractor | App manual")
        canvas.drawString(50, 25, "Offline guide | English")
        canvas.drawRightString(545, 25, str(doc.page))

    pdf = DEST / "FormExtractor_User_Manual.pdf"
    SimpleDocTemplate(
        str(pdf),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=52,
        bottomMargin=45,
        title="FormExtractor - App manual",
        author="FormExtractor",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    output = ROOT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, output / pdf.name)
    print(f"Built {DEST}")


if __name__ == "__main__":
    build()
