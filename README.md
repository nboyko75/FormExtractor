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
- **Recognition language**, with its label and selector on one line after Import, selects the language used
  by PaddleOCR. The choices match PP-StructureV3 in PaddleOCR 3.3, with native names
  and language codes. English is selected initially; subsequent launches restore the selection.
  The selection is locked while
  recognition is running; a newly selected language may require downloading models.
- **Import** sits before the language selector; **Exit** sits next to **Save** in the right pane.
- **Circle +** (Add area), at the right of the zoom toolbar, enables a crosshair. Drag over the
  scan to create a red numbered recognition rectangle. Drawings work at any zoom;
  very small accidental drags are ignored. Areas belong to the currently displayed page.
- **Circle −** (Remove area) enables a pointer with a minus badge. Click inside an area to delete
  it. Remaining area numbers stay unchanged. Press Escape or click the active mode
  button again to return to dragging/panning. On narrow windows, page navigation
  wraps below the zoom/area toolbar.
- **File mask**, above the preview on the right, selects supported files in the imported
  file's folder. Use `*.pdf`, `scan*.png`, or multiple masks such as `*.png;*.jpg`.
  Matching is case-insensitive, alphabetical, and does not recurse into subfolders.
  The default `*` matches all supported documents and images in that folder.
- **Recognize** requires at least one area. It searches for each selected area's image
  on the corresponding page of every matching file, allowing rotation, movement, and
  changes in scale. Detected text is excluded from image matching; borders and graphics
  identify the area even when its text changes. Matched crops retain their original text
  and are rotated back to the template's orientation for recognition with the selected
  language. Results appear after each area, then processing
  continues through the remaining areas and files. Unmatched areas are reported as errors
  without stopping the batch. Select distinctive areas with enough visual detail to match;
  text-only, small, or repetitive regions may not be identifiable. Recognition runs in a
  separate process, and the status bar displays the file currently being processed.
- **Stop**, immediately before Save, terminates the recognition worker without waiting
  for the current page to finish. Results already displayed remain available for saving;
  unfinished areas remain empty. Recognition can be started again immediately.
- The right pane opens with **Text** first, followed by **JSON**. Results are a table
  with one row per file and one column per area, ordered by area number. Both are editable;
  edits to each format are independent, and Save defaults to the selected tab's format.
- **Save...** provides `.txt` and `.json` filters and saves the corresponding editor. JSON is validated before saving.
- **Exit** closes the window and stops its recognition worker.

The menu provides **File → Import, Recognize, Save, Settings, Exit** and **About**.
The menu actions perform the same actions as the corresponding buttons; Recognize
and Save become available when areas and results are ready. About shows the application
version and author contact details, using the same layout as DocExplorer.

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
Area changes are saved immediately under the source file's
absolute path and restored when that file is imported again. Coordinates are stored
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
line breaks inside cells as `\n`. JSON retains original multiline values. Files should
share the template's layout; no automatic alignment between different form layouts is performed.

The CLI/Python whole-document API retains its existing output: JSON contains `source`,
`mode: "paddle"`, `model: "PP-StructureV3"`, and `pages`. Each page includes:

- `page`, `image_width`, `image_height`: page identity and processed image dimensions.
- `text`: PaddleOCR's full reconstructed Markdown, including detected table HTML.
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
