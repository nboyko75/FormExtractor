# FormExtractor

FormExtractor recognizes text in scanned PDFs and images using **PaddleOCR PP-StructureV3**, running locally on CPU by default. It includes a desktop form, a command-line interface, and a Python API. Recognition does not use Ollama or a language-model server.

## Installation

Use Python 3.10 or 3.11 for the pinned PaddleOCR 3.3.x / PaddlePaddle 3.2.x dependencies.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

This installs PaddleOCR's document parsing dependencies and the CPU PaddlePaddle runtime. On first recognition, PaddleOCR downloads its models. After the models are cached, recognition can run offline. No scans are submitted to a remote recognition service.

On Linux/macOS, use `python3 -m venv .venv` and `.venv/bin/python` instead of the Windows executable path. Runtime wheel availability depends on OS and Python version. See the [PaddleOCR installation instructions](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.3.2/docs/version3.x/installation.en.md).

## Desktop application

```powershell
.\.venv\Scripts\python.exe main.py
```

Alternatively, use `python -m formextractor.mainform` or the installed `formextractor-gui` command. Tkinter is included with standard Windows Python; Linux installations may require the distribution's Tkinter package.

- **Import...** opens a PDF, JPG, PNG, TIFF, BMP, or WebP in the left preview pane.
- **Previous / Next** browse pages of a PDF or multi-frame image.
- **Magnifying-glass + / − buttons** below the preview, before Previous / Next,
  or the mouse wheel over the preview,
  enlarge or reduce the scan from 25% to 400% of its fitted size. Use the scrollbars
  or drag the scan to move around an enlarged page. The zoom percentage is preserved
  when importing another document, browsing pages, or restarting the application.
- **Recognition language**, with its label and selector on one line below the preview, selects the language used
  by PaddleOCR. The choices match PP-StructureV3 in PaddleOCR 3.3, with native names
  and language codes. English is selected initially; subsequent launches restore the selection.
  The selection is locked while
  recognition is running; a newly selected language may require downloading models.
- **Import** sits immediately before the source file's full path above the preview. Long paths wrap
  to fit the pane. Drag the raised 2 mm divider strip to resize the preview and results panes.
  **Exit** sits next to **Save** in the right pane.
- **Text / Number / Date** groups on the left of the recognition-language row each have
  compact **+** (Add) and **−** (Remove) buttons. Add enables a crosshair; drag over the value to create a
  numbered rectangle labelled with its type. Text areas are red, numbers blue,
  and dates green. Drawings work at any zoom; very small accidental drags are ignored.
  Areas belong to the currently displayed page. Remove deletes only the chosen type
  on that page, even when areas overlap. Remaining area numbers stay unchanged.
  Press Escape or click the active button again to return to dragging/panning.
  On narrow windows, page navigation wraps below the zoom controls.
- **Number** areas restrict character decoding to digits, decimal points/commas,
  signs (`+`, `-`), and spaces. **Date** areas allow digits, `.`, `/`, `-`, and spaces.
  Select a single value per number/date area. These types use PP-OCRv5's text
  recognizer with a restricted character set before decoding; text areas continue
  to use PP-StructureV3 with the selected language. Dates are formatted as
  `DD.MM.YYYY`, removing extra punctuation and year suffixes such as `р.`.
  If restricted decoding produces an empty or invalid date, the same crop is retried with
  the normal alphabet and validated again. Successful date readings keep their path.
  Accepted formats include day-first dates with dots, slashes, hyphens or spaces,
  year-first dates (`YYYY-MM-DD`), unambiguous US dates (`03/21/2026`), compact
  `DDMMYYYY`, `YYYYMMDD`, `DDMMYY`, and recognized English/Ukrainian/Russian month
  names. Ambiguous numeric order is day-first (`06/04/2026` means 6 April).
  Two-digit years use 2000–2049 for `00`–`49` and 1950–1999 for `50`–`99`.
  Invalid dates, multiple dates and ambiguous compact dates are reported as errors.
  The numeric recognizer is downloaded on
  first use if it is not already cached. Existing saved areas default to Text;
  newly selected types are saved with their coordinates and included in JSON results.
- **File mask**, above the preview on the right, selects supported files in the imported
  file's folder. Use `*.pdf`, `scan*.png`, or multiple masks such as `*.png;*.jpg`.
  Matching is case-insensitive, alphabetical, and does not recurse into subfolders.
  The default `*` matches all supported documents and images in that folder.
- **Recognize** requires at least one area. It first aligns the surrounding form on the
  corresponding page of every matching file, allowing rotation, movement, and changes
  in scale. Photos of ruled forms also use the paper outline and internal table lines
  to correct perspective; the outline alone is never enough to accept a match.
  If a tightly cropped scan hides the paper outline and the usual alignment fails,
  a fallback uses the internal ruling to align the form. Successful existing
  alignments are kept as they are.
  All selected areas use that shared alignment, so small or repeated table
  cells keep their positions. Detected text is excluded from image matching; borders
  and graphics establish alignment even when field values change. If page alignment
  cannot be verified, individual areas are matched using their own visual detail.
  Matched crops retain their original text
  and are rotated back to the template's orientation for recognition with the selected
  language. Results appear after each area, then processing
  continues through the remaining areas and files. Unmatched areas are reported as errors
  without stopping the batch. Text-only, small, or repetitive regions need enough
  surrounding form structure to establish page alignment. Recognition runs in a
  separate process, and the status bar displays the file currently being processed.
- **Save areas**, on the right of the recognition-language row, saves the matched crops after
  each file finishes recognition. The checkbox wraps below that row in narrow panes.
  One PNG is written in an automatically created `areas` subfolder, for example `areas/invoice_areas.png`
  for `invoice.pdf`. Crops are stacked in area-number order, including matches from different
  pages, with white padding when widths differ. Unmatched regions are omitted. Matching crops
  are still saved if OCR fails for an area; failures remain in the results. The checkbox is
  remembered between sessions. Generated `*_areas.png` files are excluded from batch inputs.
  Existing output images are replaced only after the new PNG is complete; source files are
  protected, and conflicting output names are reported as errors.
- **Stop**, immediately before Save, terminates the recognition worker without waiting
  for the current page to finish. Results already displayed remain available for saving;
  unfinished areas remain empty. Recognition can be started again immediately.
- The right pane opens with **Text** first, followed by **JSON**. Results are a table
  with one row per file and one column per area, ordered by area number. Both are editable;
  edits to each format are independent, and Save defaults to the selected tab's format.
- **Save...** provides `.txt` and `.json` filters and saves the corresponding editor. JSON is validated before saving.
- **Exit** closes the window and stops its recognition worker.

The menu provides **File → Import, Recognize, Save, Settings, Exit** and **Help → App manual (F1), About**.
The menu actions perform the same actions as the corresponding buttons; Recognize
and Save become available when areas and results are ready. About shows the application
version and author contact details, using the same layout as DocExplorer. App manual
opens the bundled offline English guide in the default browser, with a printable PDF link.
The guide includes the supplied `store/Screenshots/Main_01.png` screenshot.

Read the [HTML manual](src/formextractor/manual/index.html) or
[PDF manual](src/formextractor/manual/FormExtractor_User_Manual.pdf).
To regenerate both, install the development dependencies and run
`python tools/build_manual.py` (Arial on Windows or DejaVu Sans on Linux).
The wheel and Windows executable include the HTML, PDF and screenshot.

In **File → Settings → Main**, select the interface language: English, German,
French, Spanish, Portuguese (Portugal), Italian, Ukrainian, or Russian. **OK** saves
a changed language, displays a restart warning, and restarts the desktop session.
The current document closes and any running recognition stops; unsaved result edits
must be confirmed before restarting. **Cancel**, Escape, and the window close button
discard settings changes. An unchanged language does not require a restart.

Preferences are saved per user in `%APPDATA%\FormExtractor\settings.json` on Windows,
or `$XDG_CONFIG_HOME/FormExtractor/settings.json` (default `~/.config/FormExtractor/settings.json`)
on other systems. The interface language does not change the OCR recognition language.
Window size, position, maximized state, pane divider, OCR language, file mask, and
zoom percentage are restored on startup. Splitter, mask, language-selection, and
zoom changes are saved automatically after a short delay, as well as on exit.
Area coordinates are saved immediately after changes and again before closing, under
the source file's absolute path. On startup, the last document and page reopen with their
saved areas; importing another document restores that document's own areas. If the last
document is unavailable, the form opens empty and keeps its saved areas. Coordinates are stored
relative to page dimensions, so they survive zoom changes and scale across matching files.
Native file dialogs use the operating system's language; technical library diagnostics
and recognized document contents retain their original text.

`Ctrl+O` imports a document and `Ctrl+S` saves results. Previewing does not load OCR models.

## Windows executable

With a 64-bit Python 3.10 or 3.11 environment in `.venv`, run:

```powershell
.\build.cmd
```

The script installs the project and PyInstaller, using compatible wheels in `Whl/`
when available and pip's configured index for missing packages. It builds the desktop
app at `dist/FormExtractor/FormExtractor.exe` and replaces any previous build there.
Distribute the entire `dist/FormExtractor/` folder. OCR models are downloaded on first
recognition unless already cached; the build does not bundle model weights.
Run `.\build.cmd --help` for usage.

The build includes a PaddleX hook in `packaging/hooks/` that copies metadata for
the OCR optional dependencies. PaddleX requires this metadata at runtime, even
when the package code is already bundled. If an older executable reports
"A dependency error occurred during pipeline creation", rebuild with `build.cmd`
and replace the entire application folder (and rebuild the MSIX if used).
Installing packages into `.venv` alone does not update an existing executable.

If first recognition reports a model download failure with
`AttributeError: 'NoneType' object has no attribute 'write'`, rebuild with
`build.cmd` and then rebuild/reinstall the MSIX if used. The OCR adapter now
provides writable output streams when running without a console, so PaddleX's
download progress cannot fail on a missing `stderr`. `build_dev.cmd` packages
the existing executable; it does not rebuild Python code. Model downloads still
require a working internet connection. If a model host is unavailable, PaddleX
can use an alternative source; see its
[model source settings](https://paddlepaddle.github.io/PaddleX/3.5/en/module_usage/tutorials/ocr_modules/layout_analysis.html).

If recognition reports `Cannot open file ... inference.json` even though the file
exists, a non-ASCII Windows user-folder name can prevent Paddle's native runtime
from opening the model. The app now uses the cache folder's ASCII Windows short
path before importing PaddleOCR, reusing existing downloads. Rebuild with
`build.cmd`, then rebuild/reinstall the MSIX if used. If Windows short names are
unavailable, set `PADDLE_PDX_CACHE_HOME` to a writable ASCII-only path (for example
`D:\FormExtractorModels`) before launching the app; models will download there.

To package that build as MSIX, run `.\build_dev.cmd` for a developer-signed installer
or `.\build_store.cmd` for a Partner Center upload package. Outputs are written to
`store/`. The package identity is configured as `NickBoiko.FormExtractor`;
see [MSIX packaging](store/README.md) for SDK requirements, signing, and installation.

## Command line

```powershell
# Recognized text
.\.venv\Scripts\python.exe -m formextractor form.pdf --format text -o output/result.txt

# JSON with text, simple field pairs, and full OCR evidence
.\.venv\Scripts\python.exe -m formextractor form.png -o output/result.json

# Selected pages, additional clockwise rotation, and OCR language
.\.venv\Scripts\python.exe -m formextractor form.pdf --pages 1,3-5 --rotate 90 --ocr-lang en

.\.venv\Scripts\python.exe -m formextractor --help
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--format` | `json` | `text` or `json` |
| `--output`, `-o` | stdout | UTF-8 output path |
| `--pages` | All | One-based pages, such as `1,3-5` |
| `--dpi` | `200` | PDF rendering resolution |
| `--max-pixels` | `4000000` | Maximum processed page image area |
| `--rotate` | `0` | Additional clockwise rotation: 0, 90, 180, 270 |
| `--ocr-device` | `cpu` | Paddle device; GPU needs a compatible GPU runtime |
| `--ocr-lang` | `en` | PaddleOCR recognition language code |
| `--verbose` | Off | Detailed error diagnostics |

PDF rotation and EXIF orientation are respected; transparency is flattened onto white. Pages are processed in document order. A failure on any page leaves an existing output file intact. Exit codes: `0` success, `1` recognition failure, `2` invalid CLI arguments, `130` cancellation.

## Output and limitations

Desktop batch JSON contains `source` (the template file), `mode: "areas"`, `model`,
`language`, `file_mask`, the numbered `areas` with page numbers and relative coordinates,
`columns` (ascending area numbers), and `rows`. Each row contains `file`, an ordered
`values` array, and an optional `error`. Unreadable files, failed crops, and missing
template pages are reported per file without discarding other files' results; values
that could not be recognized are `null`. The Text tab shows an aligned table, escaping
line breaks inside cells as `\n`. Recognized values omit HTML tags, retaining their
text content; table cells are separated by tabs. JSON retains multiline values. Files should
share the template's layout; no automatic alignment between different form layouts is performed.
If layout detection exports a text area as an image placeholder, an otherwise empty
value is recovered from the same prediction's OCR text. This fallback does not
replace existing nonempty text values or use unrestricted OCR for number/date areas.
Number values remove surrounding text, digit-grouping spaces, leading zeros and
redundant decimal zeros; decimal commas become dots (for example, `-001,50 km`
becomes `-1.5`). Ambiguous numeric expressions are reported as recognition errors.
Values remain strings in batch JSON, including normalized numbers and dates.

The CLI/Python whole-document API retains its existing output: JSON contains `source`,
`mode: "paddle"`, `model: "PP-StructureV3"`, and `pages`. Each page includes:

- `page`, `image_width`, `image_height`: page identity and processed image dimensions.
- `text`: Recognized text with HTML tags and leading heading markers removed.
- `fields`: simple explicit `Name: value` lines split into `number`, `name`, and `value`. Repeated labels and leading zeros are preserved. This is a heuristic; review the results.
- `ocr.markdown` and `ocr.raw`: full PaddleOCR output, including detected layout regions, text coordinates, recognition scores, and table structures where available.

Text export uses the complete recognized page text, including text that does not form a field pair. Coordinates refer to the processed page image after rotation and resizing.

PaddleOCR recognizes content and layout; arbitrary label-to-value matching is not inferred. Labels and values in separate regions remain in the OCR text. The compatibility fields `document_type` and `tables` are currently `null` and `[]`: table structure is retained in `ocr.raw` and Markdown rather than converted into the previous normalized table schema. No field-level confidence or bounding boxes are invented.

Recognition accuracy depends on scan quality and language. Small text, handwriting, merged cells, and unusual layouts need review. Pages are independent; tables spanning pages are not merged. Automatic deskew and password unlocking are not implemented.

## Python API

```python
from formextractor import extract_document
from formextractor.paddle import PaddleLayout

result = extract_document("form.pdf", layout=PaddleLayout(device="cpu", lang="en"))
for page in result.pages:
    print(page.text)
print(result.model_dump_json(indent=2))
```

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Tests use simulated Paddle predictions for integration checks; they do not measure recognition accuracy. To generate a synthetic scanned form for real OCR testing:

```powershell
.\.venv\Scripts\python.exe examples/create_sample.py
.\.venv\Scripts\python.exe -m formextractor output/sample/vehicle_form.pdf --format text
```

The sample includes `Skoda Yeti`, `2011`, `102000 km`, `00123`, and a two-row condition table. Generated files are under the ignored `output/` directory.

Reference: [PP-StructureV3 API and result format](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.3.2/docs/version3.x/pipeline_usage/PP-StructureV3.en.md).
