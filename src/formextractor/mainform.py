"""Desktop document preview and recognition results."""

import multiprocessing
import re
import sys
import tkinter as tk
from contextlib import closing
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pymupdf
from PIL import Image, ImageTk

from .area_editor import AreaEditor
from .areas import BatchResult, BatchRow, batch_text, matching_files, read_areas, source_key
from .dialogs import AboutForm, SettingsForm
from .documents import iter_pages
from .errors import ExtractionError
from .jobs import RecognitionJob
from .localization import Translator
from .ocr_languages import OCR_LANGUAGES
from .output import to_text, validate_output_path, write_output
from .schema import DocumentResult
from .settings import load_language, load_settings, save_language, update_settings
from .widgets import ToolTip, arrow_icon, circle_icon, exit_icon, import_icon, zoom_icon

MIN_ZOOM = 0.25
MAX_ZOOM = 4.0
ZOOM_STEP = 1.25


def load_preview(source: Path, number: int = 1) -> tuple[Image.Image, int]:
    """Return an owned preview image and count without keeping a document open."""
    if source.suffix.lower() == ".pdf":
        with pymupdf.open(source) as document:
            if document.needs_pass:
                raise ExtractionError("Unlock this PDF before importing it.")
            count = len(document)
    else:
        with Image.open(source) as document:
            count = getattr(document, "n_frames", 1)
    with closing(iter_pages(source, pages=str(number), dpi=120, max_pixels=2_000_000)) as pages:
        return next(pages).image.copy(), count


class MainForm:
    def __init__(
        self, root: tk.Tk, *, job_factory=RecognitionJob, preferences_path: Path | None = None
    ) -> None:
        self.root = root
        self.job_factory = job_factory
        self.preferences_path = preferences_path
        self.preferences = load_settings(preferences_path)
        self.tr = Translator(load_language(preferences_path))
        self.restart_requested = False
        self.settings_form = None
        self.about_form = None
        self.source: Path | None = None
        self.result: DocumentResult | BatchResult | None = None
        self.batch_sources = []
        self._batch_started = False
        self.preview: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.page_number = 1
        self.page_count = 0
        zoom = self.preferences.get("zoom_level", 1.0)
        self.zoom_level = (
            zoom if isinstance(zoom, (int, float)) and MIN_ZOOM <= zoom <= MAX_ZOOM else 1.0
        )
        self.reset_preview_view = True
        self.zoom_label = tk.StringVar(master=root, value="100%")
        self.ocr_choices = {f"{name} ({code})": code for code, name in OCR_LANGUAGES.items()}
        language = self.preferences.get("ocr_language", "en")
        selected_language = next(
            (label for label, code in self.ocr_choices.items() if code == language), "English (en)"
        )
        self.ocr_language = tk.StringVar(master=root, value=selected_language)
        mask = self.preferences.get("file_mask", "*")
        self.file_mask = tk.StringVar(master=root, value=mask if isinstance(mask, str) else "*")
        self.save_areas = tk.BooleanVar(
            master=root, value=self.preferences.get("save_areas") is True
        )
        self.job = None
        self.poll_id = None
        self.resize_id = None
        self.split_id = None
        self.preferences_id = None
        self.split_initialized = False
        self.status = tk.StringVar(value=self.tr("start"))
        self.filename = tk.StringVar(value=self.tr("no_document"))
        self.page_label = tk.StringVar(value=self.tr("no_pages"))

        root.title("FormExtractor")
        window = self.preferences.get("window", {})
        if not isinstance(window, dict):
            window = {}
        geometry = window.get("geometry", "1160x780+40+40")
        if not isinstance(geometry, str) or not re.fullmatch(r"\d+x\d+[+-]\d+[+-]\d+", geometry):
            geometry = "1160x780+40+40"
        self.normal_geometry = geometry
        root.geometry(geometry)
        root.minsize(820, 560)
        if window.get("state") == "zoomed":
            try:
                root.state("zoomed")
            except tk.TclError:
                pass
        root.bind("<Configure>", self._remember_geometry, add=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.menu = tk.Menu(root, tearoff=False)
        self.file_menu = tk.Menu(self.menu, tearoff=False)
        self.file_menu.add_command(
            label=self.tr("import"), command=self.import_document, accelerator="Ctrl+O"
        )
        self.file_menu.add_command(label=self.tr("recognize"), command=self.recognize)
        self.file_menu.add_command(label=self.tr("save"), command=self.save, accelerator="Ctrl+S")
        self.file_menu.add_command(label=self.tr("settings"), command=self.show_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=self.tr("exit"), command=self.close)
        self.menu.add_cascade(label=self.tr("file"), menu=self.file_menu)
        self.menu.add_command(label=self.tr("about"), command=self.show_about)
        root.configure(menu=self.menu)

        panes = ttk.Panedwindow(root, orient="horizontal")
        self.panes = panes
        panes.grid(row=0, column=0, sticky="nsew", padx=16, pady=(12, 0))
        left = ttk.Frame(panes, padding=(0, 0, 8, 0))
        right = ttk.Frame(panes, padding=(8, 0, 0, 0))
        panes.add(left, weight=1)
        panes.add(right, weight=1)
        panes.bind("<Configure>", self._schedule_initial_split)
        panes.bind("<ButtonRelease-1>", self._schedule_preferences_save, add=True)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        ttk.Label(left, text=self.tr("preview"), font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        source_header = ttk.Frame(left)
        source_header.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        source_header.columnconfigure(0, weight=1)
        ttk.Label(source_header, textvariable=self.filename, width=30).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(source_header, text=self.tr("file_mask")).grid(row=0, column=1, padx=(8, 4))
        self.mask_entry = ttk.Entry(source_header, textvariable=self.file_mask, width=12)
        self.mask_entry.grid(row=0, column=2, sticky="e")
        preview_frame = ttk.Frame(left)
        preview_frame.grid(row=2, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview_frame, background="#e8edf2", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview_frame, orient="vertical", command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.bind("<Configure>", self._schedule_preview)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.area_editor = AreaEditor(self.canvas, self._areas_changed)
        root.bind("<Escape>", lambda event: self.set_area_mode(None))
        navigation = ttk.Frame(left, padding=(0, 8))
        navigation.grid(row=3, column=0, sticky="ew")
        navigation.columnconfigure(1, weight=1)
        zoom_controls = ttk.Frame(navigation)
        zoom_controls.grid(row=0, column=0, sticky="w")
        page_controls = ttk.Frame(navigation)
        page_controls.grid(row=0, column=1, sticky="w")
        area_controls = ttk.Frame(navigation)
        area_controls.grid(row=0, column=2, sticky="e")
        self.area_icons = {
            (plus, disabled): circle_icon(root, plus=plus, disabled=disabled)
            for plus in (True, False)
            for disabled in (True, False)
        }
        self.add_area_button = ttk.Button(
            area_controls,
            image=(self.area_icons[True, False], "disabled", self.area_icons[True, True]),
            command=lambda: self.set_area_mode("add"),
        )
        self.add_area_button.pack(side="left", padx=(4, 4))
        self.remove_area_button = ttk.Button(
            area_controls,
            image=(self.area_icons[False, False], "disabled", self.area_icons[False, True]),
            command=lambda: self.set_area_mode("remove"),
        )
        self.remove_area_button.pack(side="left")
        ToolTip(self.add_area_button, self.tr("add_area"))
        ToolTip(self.remove_area_button, self.tr("remove_area"))

        def arrange_navigation(event):
            required = (
                sum(
                    frame.winfo_reqwidth()
                    for frame in (zoom_controls, page_controls, area_controls)
                )
                + 12
            )
            if event.width < required:
                page_controls.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
            else:
                page_controls.grid(row=0, column=1, columnspan=1, sticky="w", pady=0)

        navigation.bind("<Configure>", arrange_navigation)
        self.zoom_icons = {
            (plus, disabled): zoom_icon(root, plus=plus, disabled=disabled)
            for plus in (True, False)
            for disabled in (True, False)
        }
        self.zoom_in_button = ttk.Button(
            zoom_controls,
            image=(self.zoom_icons[True, False], "disabled", self.zoom_icons[True, True]),
            command=lambda: self.zoom(1),
        )
        self.zoom_in_button.pack(side="left")
        self.zoom_out_button = ttk.Button(
            zoom_controls,
            image=(self.zoom_icons[False, False], "disabled", self.zoom_icons[False, True]),
            command=lambda: self.zoom(-1),
        )
        self.zoom_out_button.pack(side="left", padx=(4, 6))
        ToolTip(self.zoom_in_button, self.tr("zoom_in"))
        ToolTip(self.zoom_out_button, self.tr("zoom_out"))
        ttk.Label(zoom_controls, textvariable=self.zoom_label).pack(side="left", padx=(0, 8))
        self.page_icons = {
            (right, disabled): arrow_icon(root, right=right, disabled=disabled)
            for right in (True, False)
            for disabled in (True, False)
        }
        self.previous_button = ttk.Button(
            page_controls,
            image=(self.page_icons[False, False], "disabled", self.page_icons[False, True]),
            command=lambda: self.turn_page(-1),
        )
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(
            page_controls,
            image=(self.page_icons[True, False], "disabled", self.page_icons[True, True]),
            command=lambda: self.turn_page(1),
        )
        self.next_button.pack(side="left", padx=4)
        ToolTip(self.previous_button, self.tr("previous"))
        ToolTip(self.next_button, self.tr("next"))
        ttk.Label(page_controls, textvariable=self.page_label).pack(side="left", padx=(2, 0))
        source_actions = ttk.Frame(left)
        source_actions.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        source_actions.columnconfigure(2, weight=1)
        self.import_icons = {
            disabled: import_icon(root, disabled=disabled) for disabled in (False, True)
        }
        self.import_button = ttk.Button(
            source_actions,
            image=(self.import_icons[False], "disabled", self.import_icons[True]),
            command=self.import_document,
        )
        self.import_button.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.import_tooltip = ToolTip(self.import_button, self.tr("import"))
        self.ocr_language_label = ttk.Label(source_actions, text=self.tr("ocr_language"))
        self.ocr_language_label.grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.ocr_language_box = ttk.Combobox(
            source_actions,
            textvariable=self.ocr_language,
            values=list(self.ocr_choices),
            state="readonly",
            width=18,
            height=18,
        )
        self.ocr_language_box.grid(row=0, column=2, sticky="ew")
        self.ocr_language_box.bind("<<ComboboxSelected>>", self._schedule_preferences_save)
        self.save_areas_checkbox = ttk.Checkbutton(
            source_actions,
            text=self.tr("save_areas"),
            variable=self.save_areas,
            command=self._schedule_preferences_save,
        )
        self.save_areas_checkbox.grid(row=0, column=3, sticky="e", padx=(8, 0))

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text=self.tr("result"), font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.text_editor = self._add_editor(self.tr("text"), wrap="none")
        self.json_editor = self._add_editor("JSON", wrap="none")
        ttk.Label(right, text=self.tr("engine")).grid(row=2, column=0, sticky="w", pady=(8, 8))
        actions = ttk.Frame(right)
        actions.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.recognize_button = ttk.Button(
            actions, text=self.tr("recognize"), command=self.recognize
        )
        self.recognize_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text=self.tr("stop"), command=self.stop_recognition)
        self.stop_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(actions, text=self.tr("save"), command=self.save)
        self.save_button.pack(side="left", padx=(0, 8))
        self.exit_icon = exit_icon(root)
        self.exit_button = ttk.Button(actions, image=self.exit_icon, command=self.close)
        self.exit_button.pack(side="left")
        ToolTip(self.exit_button, self.tr("exit"))

        footer = ttk.Frame(root, padding=(16, 12))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, width=1).grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=1, padx=(12, 0))
        root.bind("<Control-o>", lambda event: self.import_document())
        root.bind("<Control-s>", lambda event: self.save())
        self.file_mask.trace_add("write", self._schedule_preferences_save)
        self._update_controls()
        last_source = self.preferences.get("last_source")
        if isinstance(last_source, str) and last_source:
            self.load_document(
                Path(last_source),
                page_number=self.preferences.get("last_page", 1),
                report_errors=False,
            )

    def _schedule_initial_split(self, event=None) -> None:
        if self.split_initialized:
            return
        if self.split_id is not None:
            self.root.after_cancel(self.split_id)
        # Setting the sash during construction is overwritten by ttk's first
        # geometry pass. Wait until the mapped paned window has its actual size.
        self.split_id = self.root.after_idle(self._initialize_split)

    def _initialize_split(self) -> None:
        self.split_id = None
        if self.panes.winfo_ismapped() and self.panes.winfo_width() > 1:
            ratio = self.preferences.get("split_fraction", 0.5)
            if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
                ratio = 0.5
            self.panes.sashpos(0, round(self.panes.winfo_width() * ratio))
            self.split_initialized = True

    def _add_editor(self, title: str, *, wrap: str) -> tk.Text:
        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        editor = tk.Text(
            frame,
            wrap=wrap,
            font=("Consolas", 11),
            undo=True,
            padx=12,
            pady=12,
            borderwidth=0,
            highlightthickness=0,
        )
        editor.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=editor.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        editor.configure(yscrollcommand=vertical.set)
        if wrap == "none":
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=editor.xview)
            horizontal.grid(row=1, column=0, sticky="ew")
            editor.configure(xscrollcommand=horizontal.set)
        self.notebook.add(frame, text=title)
        return editor

    def _update_controls(self) -> None:
        busy = self.job is not None
        can_recognize = self.source is not None and bool(self.area_editor.areas) and not busy
        self.file_menu.entryconfigure(0, state="disabled" if busy else "normal")
        self.file_menu.entryconfigure(1, state="normal" if can_recognize else "disabled")
        self.file_menu.entryconfigure(
            2, state="normal" if self.result is not None and not busy else "disabled"
        )
        self.ocr_language_box.configure(state="disabled" if busy else "readonly")
        self.mask_entry.configure(state="disabled" if busy else "normal")
        self.area_editor.enabled = not busy
        if busy or (self.area_editor.mode == "remove" and not self.area_editor.areas):
            self.area_editor.set_mode(None)
        self._update_zoom_controls()
        for widget, enabled in [
            (self.import_button, not busy),
            (self.save_areas_checkbox, not busy),
            (self.recognize_button, can_recognize),
            (self.save_button, self.result is not None and not busy),
            (self.stop_button, busy),
            (self.previous_button, self.page_number > 1),
            (self.next_button, self.page_number < self.page_count),
            (self.add_area_button, self.source is not None and not busy),
            (self.remove_area_button, bool(self.area_editor.areas) and not busy),
        ]:
            widget.state(["!disabled"] if enabled else ["disabled"])
        for button, mode in ((self.add_area_button, "add"), (self.remove_area_button, "remove")):
            button.state(["pressed"] if self.area_editor.mode == mode else ["!pressed"])
        for editor in (self.text_editor, self.json_editor):
            editor.configure(state="normal" if self.result is not None and not busy else "disabled")

    def _confirm_discard_edits(self) -> bool:
        if any(editor.edit_modified() for editor in (self.text_editor, self.json_editor)):
            return messagebox.askyesno(self.tr("unsaved"), self.tr("discard"), parent=self.root)
        return True

    def import_document(self) -> None:
        if self.job is not None:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title=self.tr("import_title"),
            filetypes=[
                (self.tr("documents_images"), "*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                (self.tr("pdf_documents"), "*.pdf"),
                (self.tr("images"), "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
            ],
        )
        if selected:
            self.load_document(Path(selected))

    def load_document(
        self, source: Path, *, page_number: int = 1, report_errors: bool = True
    ) -> None:
        if self.job is not None or not self._confirm_discard_edits():
            return
        try:
            preview, count = load_preview(source)
            if type(page_number) is not int or not 1 <= page_number <= count:
                page_number = 1
            if page_number != 1:
                preview.close()
                preview, _ = load_preview(source, page_number)
        except Exception as exc:
            if report_errors:
                messagebox.showerror(self.tr("import_error"), self.tr.error(exc), parent=self.root)
            return
        if self.preview is not None:
            self.preview.close()
        self.source = source
        files = load_settings(self.preferences_path).get("files", {})
        template = files.get(source_key(source), {}) if isinstance(files, dict) else {}
        self.area_editor.areas = (
            read_areas(template.get("areas")) if isinstance(template, dict) else []
        )
        self.area_editor.set_mode(None)
        self.preview = preview
        self.reset_preview_view = True
        self.page_number, self.page_count = page_number, count
        self.filename.set(source.name)
        self.page_label.set(self.tr("page", number=page_number, count=count))
        self.result = None
        self.batch_sources = []
        self._set_editor(self.text_editor, "")
        self._set_editor(self.json_editor, "")
        self.status.set(self.tr("areas_ready"))
        self._update_controls()
        self._draw_preview()

    def turn_page(self, direction: int) -> None:
        number = self.page_number + direction
        if self.source is None or not 1 <= number <= self.page_count:
            return
        try:
            preview, _ = load_preview(self.source, number)
        except Exception as exc:
            messagebox.showerror(self.tr("preview_error"), self.tr.error(exc), parent=self.root)
            return
        if self.preview is not None:
            self.preview.close()
        self.preview = preview
        self.area_editor.start = self.area_editor.end = None
        self.reset_preview_view = True
        self.page_number = number
        self.page_label.set(self.tr("page", number=number, count=self.page_count))
        self._draw_preview()
        self._update_controls()

    def _schedule_preview(self, event=None) -> None:
        if self.resize_id is not None:
            self.root.after_cancel(self.resize_id)
        self.resize_id = self.root.after(80, self._draw_preview)

    def _update_zoom_controls(self) -> None:
        self.zoom_label.set(f"{self.zoom_level:.0%}")
        for button, enabled in (
            (self.zoom_in_button, self.preview is not None and self.zoom_level < MAX_ZOOM),
            (self.zoom_out_button, self.preview is not None and self.zoom_level > MIN_ZOOM),
        ):
            button.state(["!disabled"] if enabled else ["disabled"])

    def zoom(self, steps: float) -> None:
        if self.preview is None:
            self.area_editor.image_box = None
            return
        level = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom_level * ZOOM_STEP ** max(-8, min(8, steps))))
        if level != self.zoom_level:
            self.zoom_level = level
            self._draw_preview()
            self._update_zoom_controls()
            self._schedule_preferences_save()

    def _on_mousewheel(self, event) -> str:
        if getattr(event, "num", None) == 4:
            steps = 1
        elif getattr(event, "num", None) == 5:
            steps = -1
        else:
            steps = event.delta if sys.platform == "darwin" else event.delta / 120
        self.zoom(steps)
        return "break"

    def _draw_preview(self) -> None:
        if self.resize_id is not None:
            self.root.after_cancel(self.resize_id)
        self.resize_id = None
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.preview is None:
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self.canvas.create_text(
                width // 2,
                height // 2,
                text=self.tr("preview_hint"),
                justify="center",
                fill="#536477",
                font=("Segoe UI", 12),
            )
            return
        region = self.canvas.cget("scrollregion")
        old_region = [float(value) for value in str(region).split()] if region else []
        center_x = center_y = 0.5
        if not self.reset_preview_view and old_region and old_region[2] and old_region[3]:
            center_x = self.canvas.canvasx(width / 2) / old_region[2]
            center_y = self.canvas.canvasy(height / 2) / old_region[3]
        fit = min(
            max(width - 24, 1) / self.preview.width,
            max(height - 24, 1) / self.preview.height,
            1.0,
        )
        scale = fit * self.zoom_level
        size = (
            max(1, round(self.preview.width * scale)),
            max(1, round(self.preview.height * scale)),
        )
        display = self.preview.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display, master=self.root)
        display.close()
        region_width, region_height = max(width, size[0] + 24), max(height, size[1] + 24)
        self.canvas.configure(scrollregion=(0, 0, region_width, region_height))
        self.canvas.create_image(
            region_width / 2, region_height / 2, image=self.photo, anchor="center"
        )
        self.canvas.xview_moveto((center_x * region_width - width / 2) / region_width)
        self.canvas.yview_moveto((center_y * region_height - height / 2) / region_height)
        self.reset_preview_view = False
        self.area_editor.page = self.page_number
        self.area_editor.image_box = (
            (region_width - size[0]) / 2,
            (region_height - size[1]) / 2,
            *size,
        )
        self.area_editor.render()

    @staticmethod
    def _set_editor(editor: tk.Text, content: str) -> None:
        editor.configure(state="normal")
        editor.delete("1.0", "end")
        editor.insert("1.0", content)
        editor.edit_reset()
        editor.edit_modified(False)

    def recognize(self) -> None:
        if self.source is None or self.job is not None or not self._confirm_discard_edits():
            return
        if not self.area_editor.areas:
            messagebox.showwarning(
                self.tr("recognize"), self.tr("areas_required"), parent=self.root
            )
            return
        try:
            sources = matching_files(self.source, self.file_mask.get())
            if not sources:
                raise ExtractionError(self.tr("no_matches"))
            if not self._persist_settings():
                return
            self.batch_sources = sources
            self._batch_started = False
            self.job = self.job_factory(
                str(self.source),
                lang=self.ocr_choices[self.ocr_language.get()],
                areas=[area.model_dump() for area in self.area_editor.areas],
                sources=[str(source) for source in sources],
                mask=self.file_mask.get(),
                save_areas=self.save_areas.get(),
            )
            self.job.start()
        except Exception as exc:
            self._finish_job()
            messagebox.showerror(self.tr("start_error"), self.tr.error(exc), parent=self.root)
            return
        self.status.set(self.tr("recognizing_file", name=sources[0].name))
        self.progress.start(12)
        self._update_controls()
        self.poll_id = self.root.after(100, self._poll_job)

    def _poll_job(self) -> None:
        self.poll_id = None
        if self.job is None:
            return
        try:
            # Drain progress notices so a quick batch cannot leave the status
            # bar lagging behind the worker by one timer interval per file.
            for _ in range(100):
                message = self.job.poll()
                if message is None:
                    self.poll_id = self.root.after(100, self._poll_job)
                    return
                kind, content = message
                if kind == "row_result":
                    self._display_batch_row(BatchRow.model_validate_json(content))
                    # Let Tk paint each completed area before consuming more.
                    self.poll_id = self.root.after(1, self._poll_job)
                    return
                if kind != "progress":
                    break
                self.status.set(self.tr("recognizing_file", name=Path(content).name))
            else:
                self.poll_id = self.root.after(1, self._poll_job)
                return
            if kind == "error":
                raise ExtractionError(content)
            result = (
                BatchResult.model_validate_json(content)
                if kind == "batch_result"
                else DocumentResult.model_validate_json(content)
            )
            self.result = result
            self._set_editor(
                self.text_editor,
                batch_text(result, self.tr) if isinstance(result, BatchResult) else to_text(result),
            )
            self._set_editor(self.json_editor, result.model_dump_json(indent=2) + "\n")
            self.status.set(
                self.tr(
                    "batch_complete",
                    count=len(result.rows),
                    errors=sum(bool(row.error) for row in result.rows),
                )
                if isinstance(result, BatchResult)
                else self.tr("complete", count=len(result.pages))
            )
        except Exception as exc:
            self.status.set(self.tr("failed_status"))
            messagebox.showerror(self.tr("failed"), self.tr.error(exc), parent=self.root)
        finally:
            if self.poll_id is None:
                self._finish_job()

    def _display_batch_row(self, row: BatchRow) -> None:
        if not self._batch_started:
            ordered = sorted(self.area_editor.areas, key=lambda area: area.number)
            self.result = BatchResult(
                source=str(self.source),
                language=self.ocr_choices[self.ocr_language.get()],
                file_mask=self.file_mask.get(),
                areas=ordered,
                columns=[area.number for area in ordered],
                rows=[],
            )
            self._batch_started = True
        for index, existing in enumerate(self.result.rows):
            if existing.file == row.file:
                self.result.rows[index] = row
                break
        else:
            self.result.rows.append(row)
        self._set_editor(self.text_editor, batch_text(self.result, self.tr))
        self._set_editor(self.json_editor, self.result.model_dump_json(indent=2) + "\n")
        self.text_editor.configure(state="disabled")
        self.json_editor.configure(state="disabled")

    def _finish_job(self) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
        self.progress.stop()
        self._update_controls()

    def stop_recognition(self) -> None:
        if self.job is None:
            return
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        job, self.job = self.job, None
        try:
            job.stop()
        except Exception as exc:
            self.status.set(self.tr("failed_status"))
            messagebox.showerror(self.tr("stop"), self.tr.error(exc), parent=self.root)
        else:
            self.status.set(self.tr("stopped"))
        finally:
            self.progress.stop()
            self._update_controls()

    def save(self) -> None:
        if self.result is None or self.source is None or self.job is not None:
            return
        json_selected = self.notebook.select() == str(self.json_editor.master)
        types = [(self.tr("text_files"), "*.txt"), (self.tr("json_files"), "*.json")]
        if json_selected:
            types.reverse()
        extension = ".json" if json_selected else ".txt"
        selected_type = tk.StringVar(value=types[0][0])
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title=self.tr("save_title"),
            filetypes=types,
            defaultextension=extension,
            initialfile=self.source.stem + extension,
            typevariable=selected_type,
        )
        if not selected:
            return
        destination = Path(selected)
        if not destination.suffix:
            suffix = ".json" if "json" in selected_type.get().lower() else ".txt"
            destination = destination.with_suffix(suffix)
        try:
            if destination.suffix.lower() == ".json":
                editor = self.json_editor
                # Keep edited JSON compatible with the application's output contract.
                value = type(self.result).model_validate_json(editor.get("1.0", "end-1c"))
                content = value.model_dump_json(indent=2) + "\n"
            elif destination.suffix.lower() == ".txt":
                editor = self.text_editor
                content = editor.get("1.0", "end-1c")
            else:
                raise ExtractionError("Choose a .txt or .json filename.")
            for source in self.batch_sources:
                validate_output_path(destination, source)
            write_output(destination, content, source=self.source)
        except Exception as exc:
            messagebox.showerror(self.tr("save_error"), self.tr.error(exc), parent=self.root)
            return
        editor.edit_modified(False)
        self.status.set(self.tr("saved", name=destination.name))

    def show_settings(self) -> None:
        if self.settings_form is not None and self.settings_form.window.winfo_exists():
            self.settings_form.window.lift()
            return
        self.settings_form = SettingsForm(self.root, self.tr, self.apply_language)

    def apply_language(self, language: str) -> None:
        if language == self.tr.language:
            self.settings_form.window.destroy()
            self.settings_form = None
            return
        if not self._confirm_discard_edits():
            return
        if not self._persist_settings():
            return
        try:
            save_language(language, self.preferences_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                self.tr("settings_error"), self.tr.error(exc), parent=self.settings_form.window
            )
            return
        messagebox.showwarning(
            self.tr("restart_title"), self.tr("restart_message"), parent=self.settings_form.window
        )
        self.restart_requested = True
        self._shutdown()

    def show_about(self) -> None:
        if self.about_form is not None and self.about_form.window.winfo_exists():
            self.about_form.window.lift()
            return
        self.about_form = AboutForm(self.root, self.tr)

    def close(self) -> None:
        if not self._confirm_discard_edits():
            return
        if not self._persist_settings():
            return
        self._shutdown()

    def _remember_geometry(self, event) -> None:
        if event.widget == self.root and self.root.state() == "normal":
            if self.root.winfo_width() >= 820 and self.root.winfo_height() >= 560:
                self.normal_geometry = self.root.geometry()

    def _persist_settings(self) -> bool:
        changes = {
            "window": {
                "geometry": self.normal_geometry,
                "state": "zoomed" if self.root.state() == "zoomed" else "normal",
            },
            "ocr_language": self.ocr_choices.get(self.ocr_language.get(), "en"),
            "file_mask": self.file_mask.get(),
            "save_areas": self.save_areas.get(),
            "zoom_level": self.zoom_level,
        }
        if self.split_initialized and self.panes.winfo_width() > 1:
            changes["split_fraction"] = self.panes.sashpos(0) / self.panes.winfo_width()
        if self.source is not None:
            changes["last_source"] = str(self.source.resolve())
            changes["last_page"] = self.page_number
            files = load_settings(self.preferences_path).get("files", {})
            if not isinstance(files, dict):
                files = {}
            files[source_key(self.source)] = {
                "areas": [area.model_dump() for area in self.area_editor.areas]
            }
            changes["files"] = files
        try:
            update_settings(changes, self.preferences_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(self.tr("settings_error"), self.tr.error(exc), parent=self.root)
            return False
        return True

    def _schedule_preferences_save(self, *args) -> None:
        if self.preferences_id is not None:
            self.root.after_cancel(self.preferences_id)
        self.preferences_id = self.root.after(350, self._save_preferences_idle)

    def _save_preferences_idle(self) -> None:
        self.preferences_id = None
        self._persist_settings()

    def _areas_changed(self) -> None:
        self._persist_settings()
        self._update_controls()

    def set_area_mode(self, mode) -> None:
        if self.source is None or self.job is not None:
            return
        self.area_editor.set_mode(None if self.area_editor.mode == mode else mode)
        self._update_controls()

    def _shutdown(self) -> None:
        for callback in (self.poll_id, self.resize_id, self.split_id, self.preferences_id):
            if callback is not None:
                self.root.after_cancel(callback)
        self._finish_job()
        if self.preview is not None:
            self.preview.close()
        self.root.quit()
        self.root.destroy()


def main() -> int:
    multiprocessing.freeze_support()
    # Recreate the entire desktop session after settings change. Keeping the
    # launcher process works for source, GUI entry points, and frozen MSIX builds.
    while True:
        root = tk.Tk()
        form = MainForm(root)
        root.mainloop()
        if not form.restart_requested:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
