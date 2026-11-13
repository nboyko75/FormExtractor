"""Desktop document preview and recognition results."""

import multiprocessing
import sys
import tkinter as tk
from contextlib import closing
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pymupdf
from PIL import Image, ImageTk

from .dialogs import AboutForm, SettingsForm
from .documents import iter_pages
from .errors import ExtractionError
from .jobs import RecognitionJob
from .localization import Translator
from .ocr_languages import OCR_LANGUAGES
from .output import to_text, write_output
from .schema import DocumentResult
from .settings import load_language, save_language

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
        self.tr = Translator(load_language(preferences_path))
        self.restart_requested = False
        self.settings_form = None
        self.about_form = None
        self.source: Path | None = None
        self.result: DocumentResult | None = None
        self.preview: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.page_number = 1
        self.page_count = 0
        self.zoom_level = 1.0
        self.reset_preview_view = True
        self.zoom_label = tk.StringVar(master=root, value="100%")
        self.ocr_language = tk.StringVar(master=root, value="English (en)")
        self.ocr_choices = {f"{name} ({code})": code for code, name in OCR_LANGUAGES.items()}
        self.job = None
        self.poll_id = None
        self.resize_id = None
        self.split_id = None
        self.split_initialized = False
        self.status = tk.StringVar(value=self.tr("start"))
        self.filename = tk.StringVar(value=self.tr("no_document"))
        self.page_label = tk.StringVar(value=self.tr("no_pages"))

        root.title("FormExtractor")
        root.geometry("1160x780")
        root.minsize(820, 560)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

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

        header = ttk.Frame(root, padding=(16, 12))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="FormExtractor", font=("Segoe UI", 18, "bold")).pack(side="left")
        tagline = ttk.Label(header, text=self.tr("tagline"), wraplength=500)
        tagline.pack(side="left", padx=20)
        header.bind(
            "<Configure>", lambda event: tagline.configure(wraplength=max(200, event.width - 250))
        )

        panes = ttk.Panedwindow(root, orient="horizontal")
        self.panes = panes
        panes.grid(row=1, column=0, sticky="nsew", padx=16)
        left = ttk.Frame(panes, padding=(0, 0, 8, 0))
        right = ttk.Frame(panes, padding=(8, 0, 0, 0))
        panes.add(left, weight=1)
        panes.add(right, weight=1)
        panes.bind("<Configure>", self._schedule_initial_split)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)
        ttk.Label(left, text=self.tr("preview"), font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(left, textvariable=self.filename, width=1).grid(
            row=1, column=0, sticky="ew", pady=(0, 8)
        )
        zoom_controls = ttk.Frame(left)
        zoom_controls.grid(row=2, column=0, sticky="w", pady=(0, 8))
        self.zoom_in_button = ttk.Button(
            zoom_controls, text=self.tr("zoom_in"), command=lambda: self.zoom(1)
        )
        self.zoom_in_button.pack(side="left")
        self.zoom_out_button = ttk.Button(
            zoom_controls, text=self.tr("zoom_out"), command=lambda: self.zoom(-1)
        )
        self.zoom_out_button.pack(side="left", padx=6)
        ttk.Label(zoom_controls, textvariable=self.zoom_label).pack(side="left")

        preview_frame = ttk.Frame(left)
        preview_frame.grid(row=3, column=0, sticky="nsew")
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
        self.canvas.bind("<ButtonPress-1>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind(
            "<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1)
        )
        navigation = ttk.Frame(left, padding=(0, 8))
        navigation.grid(row=4, column=0, sticky="ew")
        self.previous_button = ttk.Button(
            navigation, text=self.tr("previous"), command=lambda: self.turn_page(-1)
        )
        self.previous_button.pack(side="left")
        ttk.Label(navigation, textvariable=self.page_label).pack(side="left", padx=12)
        self.next_button = ttk.Button(
            navigation, text=self.tr("next"), command=lambda: self.turn_page(1)
        )
        self.next_button.pack(side="left")
        source_actions = ttk.Frame(left)
        source_actions.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        source_actions.columnconfigure(0, weight=1)
        ttk.Label(source_actions, text=self.tr("ocr_language")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.ocr_language_box = ttk.Combobox(
            source_actions,
            textvariable=self.ocr_language,
            values=list(self.ocr_choices),
            state="readonly",
            width=16,
            height=18,
        )
        self.ocr_language_box.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.import_button = ttk.Button(
            source_actions, text=self.tr("import"), command=self.import_document
        )
        self.import_button.grid(row=1, column=1, padx=(0, 8))
        self.exit_button = ttk.Button(source_actions, text=self.tr("exit"), command=self.close)
        self.exit_button.grid(row=1, column=2)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text=self.tr("result"), font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.text_editor = self._add_editor(self.tr("text"), wrap="word")
        self.json_editor = self._add_editor("JSON", wrap="none")
        edit_hint = ttk.Label(right, text=self.tr("separate_edits"), wraplength=350)
        edit_hint.grid(row=2, column=0, sticky="w", pady=(8, 4))
        right.bind(
            "<Configure>", lambda event: edit_hint.configure(wraplength=max(120, event.width - 16))
        )
        ttk.Label(right, text=self.tr("engine")).grid(row=3, column=0, sticky="w", pady=(0, 8))
        actions = ttk.Frame(right)
        actions.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self.recognize_button = ttk.Button(
            actions, text=self.tr("recognize"), command=self.recognize
        )
        self.recognize_button.pack(side="left")
        self.save_button = ttk.Button(actions, text=self.tr("save"), command=self.save)
        self.save_button.pack(side="left", padx=8)

        footer = ttk.Frame(root, padding=(16, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, width=1).grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=1, padx=(12, 0))
        root.bind("<Control-o>", lambda event: self.import_document())
        root.bind("<Control-s>", lambda event: self.save())
        self._update_controls()

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
            self.panes.sashpos(0, self.panes.winfo_width() // 2)
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
        self.file_menu.entryconfigure(0, state="disabled" if busy else "normal")
        self.file_menu.entryconfigure(
            1, state="normal" if self.source is not None and not busy else "disabled"
        )
        self.file_menu.entryconfigure(
            2, state="normal" if self.result is not None and not busy else "disabled"
        )
        self.ocr_language_box.configure(state="disabled" if busy else "readonly")
        self._update_zoom_controls()
        for widget, enabled in [
            (self.import_button, not busy),
            (self.recognize_button, self.source is not None and not busy),
            (self.save_button, self.result is not None and not busy),
            (self.previous_button, self.page_number > 1),
            (self.next_button, self.page_number < self.page_count),
        ]:
            widget.state(["!disabled"] if enabled else ["disabled"])
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

    def load_document(self, source: Path) -> None:
        if self.job is not None or not self._confirm_discard_edits():
            return
        try:
            preview, count = load_preview(source)
        except Exception as exc:
            messagebox.showerror(self.tr("import_error"), self.tr.error(exc), parent=self.root)
            return
        if self.preview is not None:
            self.preview.close()
        self.source = source
        self.preview = preview
        self.zoom_level = 1.0
        self.reset_preview_view = True
        self.page_number, self.page_count = 1, count
        self.filename.set(source.name)
        self.page_label.set(self.tr("page", number=1, count=count))
        self.result = None
        self._set_editor(self.text_editor, "")
        self._set_editor(self.json_editor, "")
        self.status.set(self.tr("ready"))
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
            return
        level = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom_level * ZOOM_STEP ** max(-8, min(8, steps))))
        if level != self.zoom_level:
            self.zoom_level = level
            self._draw_preview()
            self._update_zoom_controls()

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
        try:
            self.job = self.job_factory(
                str(self.source), lang=self.ocr_choices[self.ocr_language.get()]
            )
            self.job.start()
        except Exception as exc:
            self._finish_job()
            messagebox.showerror(self.tr("start_error"), self.tr.error(exc), parent=self.root)
            return
        self.status.set(self.tr("recognizing"))
        self.progress.start(12)
        self._update_controls()
        self.poll_id = self.root.after(100, self._poll_job)

    def _poll_job(self) -> None:
        self.poll_id = None
        if self.job is None:
            return
        try:
            message = self.job.poll()
            if message is None:
                self.poll_id = self.root.after(100, self._poll_job)
                return
            kind, content = message
            if kind == "error":
                raise ExtractionError(content)
            result = DocumentResult.model_validate_json(content)
            self.result = result
            self._set_editor(self.text_editor, to_text(result))
            self._set_editor(self.json_editor, result.model_dump_json(indent=2) + "\n")
            self.status.set(self.tr("complete", count=len(result.pages)))
        except Exception as exc:
            self.status.set(self.tr("failed_status"))
            messagebox.showerror(self.tr("failed"), self.tr.error(exc), parent=self.root)
        finally:
            if self.poll_id is None:
                self._finish_job()

    def _finish_job(self) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
        self.progress.stop()
        self._update_controls()

    def save(self) -> None:
        if self.result is None or self.source is None or self.job is not None:
            return
        json_selected = self.notebook.index(self.notebook.select()) == 1
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
                value = DocumentResult.model_validate_json(editor.get("1.0", "end-1c"))
                content = value.model_dump_json(indent=2) + "\n"
            elif destination.suffix.lower() == ".txt":
                editor = self.text_editor
                content = editor.get("1.0", "end-1c")
            else:
                raise ExtractionError("Choose a .txt or .json filename.")
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
        self._shutdown()

    def _shutdown(self) -> None:
        for callback in (self.poll_id, self.resize_id, self.split_id):
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
