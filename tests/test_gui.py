import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

tk = pytest.importorskip("tkinter")

from formextractor.mainform import MainForm, load_preview  # noqa: E402
from formextractor.schema import DocumentResult, PageResult  # noqa: E402


@pytest.fixture(scope="module")
def desktop():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"A desktop display is required for GUI tests: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def form(monkeypatch, desktop, tmp_path):
    root = tk.Toplevel(desktop)
    root.withdraw()
    errors = []
    monkeypatch.setattr(
        "formextractor.mainform.messagebox.showerror", lambda *a, **k: errors.append(a)
    )
    monkeypatch.setattr("formextractor.mainform.messagebox.askyesno", lambda *a, **k: True)
    window = MainForm(root, preferences_path=tmp_path / "settings.json")
    root.update()
    window.test_errors = errors
    yield window
    if root.winfo_exists():
        window.close()


@pytest.fixture
def result(image_path, page_extraction):
    return DocumentResult(
        source=str(image_path),
        mode="paddle",
        model="test-model",
        pages=[
            PageResult(**page_extraction.model_dump(), page=1, image_width=120, image_height=80)
        ],
    )


class FakeJob:
    def __init__(self, source, *, lang="en"):
        self.source = source
        self.lang = lang
        self.message = None
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def poll(self):
        return self.message

    def close(self):
        self.closed = True


def recognize_result(form, image_path, result):
    form.load_document(image_path)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    job.message = ("result", result.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    return job


def test_import_button_preview_and_initial_states(form, image_path, monkeypatch):
    assert form.notebook.tab(0, "text") == "JSON"
    assert form.notebook.select() == str(form.json_editor.master)
    assert form.recognize_button.instate(["disabled"])
    assert form.save_button.instate(["disabled"])
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.askopenfilename", lambda **k: str(image_path)
    )
    form.import_button.invoke()
    assert form.source == image_path
    assert form.preview.size == (120, 80)
    assert form.page_label.get() == "Page 1 of 1"
    assert form.recognize_button.instate(["!disabled"])
    assert form.save_button.instate(["disabled"])


def test_multipage_preview_navigation(form, tmp_path):
    source = tmp_path / "pages.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=300)
        document.new_page(width=300, height=200)
        document.save(source)
    form.load_document(source)
    form.next_button.invoke()
    assert form.page_number == 2
    assert form.preview.width > form.preview.height
    assert form.next_button.instate(["disabled"])
    form.previous_button.invoke()
    assert form.page_number == 1


def test_recognition_disables_duplicate_requests_and_delivers_result(form, image_path, result):
    form.load_document(image_path)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    assert job.started and job.source == str(image_path)
    assert form.import_button.instate(["disabled"])
    assert form.recognize_button.instate(["disabled"])
    form.recognize()
    assert form.job is job
    job.message = ("result", result.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert job.closed
    assert form.job is None
    assert "Identifier: 00123" in form.text_editor.get("1.0", "end")
    assert json.loads(form.json_editor.get("1.0", "end"))["pages"][0]["page"] == 1
    assert form.save_button.instate(["!disabled"])
    assert not form.text_editor.edit_modified()


def test_worker_error_recovers_controls(form, image_path):
    form.load_document(image_path)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    job.message = ("error", "PaddleOCR is not installed")
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert form.test_errors[-1] == ("Recognition failed", "PaddleOCR is not installed")
    assert form.recognize_button.instate(["!disabled"])
    assert form.save_button.instate(["disabled"])
    assert job.closed


def test_recognition_displays_unpaired_paddle_text(form, image_path, result):
    result.pages[0].text = "Unpaired heading\nRecognized text without a field label"
    recognize_result(form, image_path, result)
    text = form.text_editor.get("1.0", "end-1c")
    assert "Recognized text without a field label" in text
    assert "Unpaired heading" in text


@pytest.mark.parametrize("suffix", [".txt", ".json"])
def test_save_formats_and_edits(form, monkeypatch, image_path, result, tmp_path, suffix):
    recognize_result(form, image_path, result)
    editor = form.json_editor if suffix == ".json" else form.text_editor
    form.notebook.select(editor.master)
    destination = tmp_path / ("result" + suffix)
    dialogs = []

    def choose(**kwargs):
        dialogs.append(kwargs)
        return str(destination)

    monkeypatch.setattr("formextractor.mainform.filedialog.asksaveasfilename", choose)
    form.text_editor.insert("end", "\nReviewed: Україна")
    form.save()
    assert {pattern for _, pattern in dialogs[0]["filetypes"]} == {"*.txt", "*.json"}
    assert dialogs[0]["defaultextension"] == suffix
    assert dialogs[0]["filetypes"][0][1] == "*" + suffix
    if suffix == ".txt":
        assert "Reviewed: Україна" in destination.read_text(encoding="utf-8")
        assert not form.text_editor.edit_modified()
    else:
        assert DocumentResult.model_validate_json(destination.read_text(encoding="utf-8")) == result


def test_invalid_json_does_not_replace_saved_file(form, monkeypatch, image_path, result, tmp_path):
    recognize_result(form, image_path, result)
    destination = tmp_path / "result.json"
    destination.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.asksaveasfilename", lambda **kwargs: str(destination)
    )
    form.json_editor.delete("1.0", "end")
    form.json_editor.insert("1.0", "invalid JSON")
    form.save()
    assert destination.read_text(encoding="utf-8") == "previous"
    assert form.test_errors[-1][0] == "Cannot save result"


def test_bad_import_preserves_current_document_and_results(form, image_path, result, tmp_path):
    recognize_result(form, image_path, result)
    form.load_document(tmp_path / "missing.png")
    assert form.source == image_path
    assert form.result == result
    assert form.test_errors[-1][0] == "Cannot import document"


def test_declining_discard_keeps_edited_results(form, monkeypatch, image_path, result):
    recognize_result(form, image_path, result)
    form.text_editor.insert("end", "correction")
    monkeypatch.setattr("formextractor.mainform.messagebox.askyesno", lambda *a, **k: False)
    form.recognize()
    assert form.job is None
    assert form.text_editor.get("1.0", "end-1c").endswith("correction")


def test_exit_stops_background_job(desktop, image_path):
    root = tk.Toplevel(desktop)
    root.withdraw()
    form = MainForm(root, job_factory=FakeJob)
    root.update()
    form.load_document(image_path)
    form.recognize()
    job = form.job
    form.close()
    assert job.closed


def test_preview_image_survives_document_close(image_path):
    image, count = load_preview(Path(image_path))
    assert count == 1
    assert image.getpixel((0, 0)) == (255, 255, 255)
    image.close()


def test_both_panes_visible_after_startup_and_resize(form):
    form.root.deiconify()
    form.root.update()
    left, right = [form.root.nametowidget(name) for name in form.panes.panes()]
    assert left.winfo_ismapped() and right.winfo_ismapped()
    assert form.canvas.winfo_ismapped()
    assert form.import_button.winfo_ismapped()
    assert abs(form.panes.sashpos(0) - form.panes.winfo_width() // 2) <= 5

    form.root.geometry("820x560")
    form.root.update()
    assert left.winfo_width() > 250
    assert right.winfo_width() > 250
    assert form.import_button.winfo_ismapped()

    # A later resize must not reset a divider position chosen by the user.
    form.panes.sashpos(0, 300)
    form.root.update()
    assert form.panes.sashpos(0) == 300
    form.root.geometry("920x660")
    form.root.update()
    assert abs(form.panes.sashpos(0) - form.panes.winfo_width() // 2) > 50


def test_file_menu_import_and_exit(form, image_path, monkeypatch):
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.askopenfilename", lambda **k: str(image_path)
    )
    form.file_menu.invoke(0)
    assert form.source == image_path
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    assert form.file_menu.entrycget(0, "state") == "disabled"
    form.file_menu.invoke(5)
    assert job.closed
    assert not form.root.winfo_exists()


@pytest.mark.parametrize("action", ["button", "window", "escape"])
def test_settings_close_discards_selection(form, action):
    form.root.deiconify()
    form.root.update()
    form.file_menu.invoke(3)
    dialog = form.settings_form
    dialog.language_box.current(1)
    if action == "button":
        dialog.cancel_button.invoke()
    elif action == "window":
        dialog.window.tk.call(dialog.window.protocol("WM_DELETE_WINDOW"))
    else:
        dialog.window.update()
        dialog.window.focus_force()
        dialog.window.event_generate("<Escape>")
    assert not dialog.window.winfo_exists()
    assert not form.preferences_path.exists()
    assert not form.restart_requested
    form.show_settings()
    assert form.settings_form.language.get() == "English"


def test_settings_unchanged_ok_does_not_restart(form, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "formextractor.mainform.messagebox.showwarning", lambda *a, **k: warnings.append(a)
    )
    form.show_settings()
    dialog = form.settings_form
    dialog.ok_button.invoke()
    assert not dialog.window.winfo_exists()
    assert not form.restart_requested
    assert warnings == []


def test_settings_save_warns_and_stops_worker(form, image_path, monkeypatch):
    from formextractor.settings import load_language

    warnings = []
    monkeypatch.setattr(
        "formextractor.mainform.messagebox.showwarning", lambda *a, **k: warnings.append(a)
    )
    form.load_document(image_path)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    form.show_settings()
    form.settings_form.language_box.current(6)
    form.settings_form.ok_button.invoke()
    assert load_language(form.preferences_path) == "uk"
    assert warnings[0][0] == "Restart required"
    assert form.restart_requested
    assert job.closed
    assert not form.root.winfo_exists()


def test_settings_declined_discard_keeps_session(form, image_path, result, monkeypatch):
    recognize_result(form, image_path, result)
    form.text_editor.insert("end", "correction")
    monkeypatch.setattr("formextractor.mainform.messagebox.askyesno", lambda *a, **k: False)
    form.show_settings()
    form.settings_form.language_box.current(1)
    form.settings_form.ok_button.invoke()
    assert form.settings_form.window.winfo_exists()
    assert not form.preferences_path.exists()
    assert not form.restart_requested
    assert form.text_editor.get("1.0", "end-1c").endswith("correction")
    form.text_editor.edit_modified(False)


def test_settings_write_failure_keeps_dialog_and_session(form, monkeypatch):
    def fail(*args):
        raise PermissionError("Read-only settings directory")

    monkeypatch.setattr("formextractor.mainform.save_language", fail)
    form.show_settings()
    form.settings_form.language_box.current(1)
    form.settings_form.ok_button.invoke()
    assert form.test_errors[-1][0] == "Cannot save settings"
    assert form.settings_form.window.winfo_exists()
    assert not form.restart_requested


@pytest.mark.parametrize("language", ["en", "de", "fr", "es", "pt", "it", "uk", "ru"])
def test_localized_desktop_dialogs_and_file_filters(desktop, tmp_path, monkeypatch, language):
    from formextractor.localization import Translator
    from formextractor.settings import LANGUAGES, save_language

    path = tmp_path / "settings.json"
    save_language(language, path)
    root = tk.Toplevel(desktop)
    root.withdraw()
    window = MainForm(root, preferences_path=path)
    tr = Translator(language)
    try:
        assert window.menu.entrycget(0, "label") == tr("file")
        assert window.import_button.cget("text") == tr("import")
        assert window.status.get() == tr("start")
        dialogs = []
        monkeypatch.setattr(
            "formextractor.mainform.filedialog.askopenfilename",
            lambda **kwargs: dialogs.append(kwargs),
        )
        window.import_document()
        assert dialogs[0]["filetypes"][0][0] == tr("documents_images")
        window.show_settings()
        assert window.settings_form.language.get() == LANGUAGES[language]
        assert window.settings_form.cancel_button.cget("text") == tr("cancel")
        window.settings_form.close()
        window.show_about()
        assert window.about_form.window.title() == tr("about")
        assert window.about_form.version_label.cget("text").startswith(tr("version", version=""))
        window.about_form.close_button.invoke()
    finally:
        window.close()


def test_main_restarts_with_saved_language(tmp_path, desktop):
    # A standalone app has one Tk interpreter. Isolate it from the suite's
    # persistent desktop root and its Tcl callbacks, which share an event queue.
    script = """
import sys
from pathlib import Path
import formextractor.mainform as module

sessions = []
errors = []
original_form = module.MainForm
module.messagebox.showwarning = lambda *a, **k: None

def create(root):
    window = original_form(root, preferences_path=Path(sys.argv[1]))
    sessions.append(window)
    def interact():
        try:
            if len(sessions) == 1:
                window.show_settings()
                window.settings_form.language_box.current(2)
                window.settings_form.apply()
            else:
                assert window.tr.language == 'fr'
                assert window.file_menu.entrycget(0, 'label') == 'Importer...'
                window.close()
        except Exception as exc:
            errors.append(repr(exc))
            root.quit()
            root.destroy()
    root.after(10, interact)
    return window

module.MainForm = create
assert module.main() == 0
assert not errors, errors
assert len(sessions) == 2
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "settings.json")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_zoom_buttons_wheel_and_limits(form, image_path):
    from formextractor.mainform import MAX_ZOOM, MIN_ZOOM

    assert form.zoom_in_button.instate(["disabled"])
    assert form.zoom_out_button.instate(["disabled"])
    form.root.deiconify()
    form.root.update()
    form.load_document(image_path)
    original_size = (form.photo.width(), form.photo.height())
    source_pixels = form.preview.tobytes()
    form.zoom_in_button.invoke()
    assert form.photo.width() > original_size[0]
    assert form.zoom_label.get() == "125%"
    form.zoom_out_button.invoke()
    assert (form.photo.width(), form.photo.height()) == original_size
    form.canvas.event_generate("<MouseWheel>", delta=120)
    assert form.zoom_level == 1.25
    form.canvas.event_generate("<MouseWheel>", delta=-120)
    assert form.zoom_level == 1.0
    form.canvas.event_generate("<Button-4>")
    assert form.zoom_level == 1.25
    form.canvas.event_generate("<Button-5>")
    assert form.zoom_level == 1.0
    for _ in range(12):
        form.zoom(1)
    assert form.zoom_level == MAX_ZOOM
    assert form.zoom_in_button.instate(["disabled"])
    for _ in range(20):
        form.zoom(-1)
    assert form.zoom_level == MIN_ZOOM
    assert form.zoom_out_button.instate(["disabled"])
    assert form.preview.tobytes() == source_pixels


def test_zoomed_preview_scrolls_and_resets_for_new_document(form, tmp_path):
    from PIL import Image

    source = tmp_path / "large.png"
    Image.new("RGB", (1000, 1400), "white").save(source)
    form.root.deiconify()
    form.root.update()
    form.load_document(source)
    form.zoom(6)
    assert form.photo.height() > form.canvas.winfo_height()
    assert form.canvas.yview()[1] - form.canvas.yview()[0] < 1
    form.canvas.yview_moveto(1)
    assert form.canvas.yview()[1] == pytest.approx(1, abs=0.01)
    form.canvas.xview_moveto(1)
    assert form.canvas.xview()[1] == pytest.approx(1, abs=0.01)
    form.load_document(source)
    assert form.zoom_level == 1
    assert form.zoom_label.get() == "100%"
    assert form.canvas.yview() == (0.0, 1.0)


def test_recognize_and_save_menu_use_selected_language(
    form, image_path, result, monkeypatch, tmp_path
):
    assert form.file_menu.entrycget(1, "label") == "Recognize"
    assert form.file_menu.entrycget(2, "label") == "Save..."
    assert form.file_menu.entrycget(1, "state") == "disabled"
    assert form.file_menu.entrycget(2, "state") == "disabled"
    form.load_document(image_path)
    choice = next(label for label, code in form.ocr_choices.items() if code == "uk")
    form.ocr_language.set(choice)
    form.job_factory = FakeJob
    form.file_menu.invoke(1)
    job = form.job
    assert job.started and job.lang == "uk"
    assert form.ocr_language_box.instate(["disabled"])
    assert form.file_menu.entrycget(1, "state") == "disabled"
    assert form.file_menu.entrycget(2, "state") == "disabled"
    job.message = ("result", result.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert form.ocr_language_box.instate(["readonly", "!disabled"])
    assert form.file_menu.entrycget(1, "state") == "normal"
    assert form.file_menu.entrycget(2, "state") == "normal"
    destination = tmp_path / "result.txt"
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.asksaveasfilename", lambda **kwargs: str(destination)
    )
    form.file_menu.invoke(2)
    assert "Identifier: 00123" in destination.read_text(encoding="utf-8")


def test_source_actions_and_zoom_layout(form):
    form.root.deiconify()
    form.root.geometry("820x560")
    form.root.update()
    left = form.root.nametowidget(form.panes.panes()[0])
    assert form.import_button.master.master == left
    assert form.exit_button.master == form.save_button.master
    assert form.exit_button.winfo_rootx() > form.save_button.winfo_rootx()
    assert form.import_button.winfo_rootx() < form.ocr_language_box.winfo_rootx()
    assert form.ocr_language_box.cget("width") == 8
    assert form.zoom_in_button.winfo_rooty() > (
        form.canvas.winfo_rooty() + form.canvas.winfo_height()
    )
    assert form.zoom_out_button.winfo_rootx() < form.previous_button.winfo_rootx()
    assert not form.zoom_in_button.cget("text")
    assert form.zoom_in_button.cget("image")
