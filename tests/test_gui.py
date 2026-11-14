import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

tk = pytest.importorskip("tkinter")

from formextractor.areas import RecognitionArea  # noqa: E402
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
        window._shutdown()


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
    def __init__(self, source, *, lang="en", **options):
        self.source = source
        self.lang = lang
        self.options = options
        self.message = None
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def poll(self):
        return self.message

    def close(self):
        self.closed = True

    def stop(self):
        self.closed = True


def recognize_result(form, image_path, result):
    form.load_document(image_path)
    add_test_area(form)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    job.message = ("result", result.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    return job


def add_test_area(form):
    form.area_editor.areas = [RecognitionArea(number=1, page=1, x1=0.0, y1=0.0, x2=1.0, y2=1.0)]
    form._update_controls()


def test_import_button_preview_and_initial_states(form, image_path, monkeypatch):
    assert form.notebook.tab(0, "text") == "Text"
    assert form.notebook.select() == str(form.text_editor.master)
    assert form.recognize_button.instate(["disabled"])
    assert form.save_button.instate(["disabled"])
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.askopenfilename", lambda **k: str(image_path)
    )
    form.import_button.invoke()
    assert form.source == image_path
    assert form.preview.size == (120, 80)
    assert form.page_label.get() == "Page 1 of 1"
    assert form.recognize_button.instate(["disabled"])
    add_test_area(form)
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
    add_test_area(form)
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
    add_test_area(form)
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


def test_exit_stops_background_job(desktop, image_path, tmp_path):
    root = tk.Toplevel(desktop)
    root.withdraw()
    form = MainForm(root, job_factory=FakeJob, preferences_path=tmp_path / "settings.json")
    root.update()
    form.load_document(image_path)
    add_test_area(form)
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
    add_test_area(form)
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
    add_test_area(form)
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
    original_settings = form.preferences_path.read_text(encoding="utf-8")
    monkeypatch.setattr("formextractor.mainform.messagebox.askyesno", lambda *a, **k: False)
    form.show_settings()
    form.settings_form.language_box.current(1)
    form.settings_form.ok_button.invoke()
    assert form.settings_form.window.winfo_exists()
    assert form.preferences_path.read_text(encoding="utf-8") == original_settings
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
        assert window.import_button.cget("text") == ""
        assert window.import_button.cget("image")
        assert window.import_tooltip.text == tr("import")
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


def test_zoomed_preview_scrolls_and_keeps_zoom_for_new_document(form, tmp_path):
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
    zoom = form.zoom_level
    form.load_document(source)
    assert form.zoom_level == zoom
    assert form.zoom_label.get() == f"{zoom:.0%}"
    assert sum(form.canvas.yview()) / 2 == pytest.approx(0.5, abs=0.01)


def test_recognize_and_save_menu_use_selected_language(
    form, image_path, result, monkeypatch, tmp_path
):
    assert form.file_menu.entrycget(1, "label") == "Recognize"
    assert form.file_menu.entrycget(2, "label") == "Save..."
    assert form.file_menu.entrycget(1, "state") == "disabled"
    assert form.file_menu.entrycget(2, "state") == "disabled"
    form.load_document(image_path)
    add_test_area(form)
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
    assert form.ocr_language_label.grid_info()["row"] == form.ocr_language_box.grid_info()["row"]
    assert form.stop_button.winfo_rootx() < form.exit_button.winfo_rootx()
    assert form.stop_button.winfo_rootx() < form.save_button.winfo_rootx()
    assert form.save_areas_checkbox.master == form.import_button.master
    assert form.save_areas_checkbox.grid_info()["row"] == form.import_button.grid_info()["row"]
    assert form.save_areas_checkbox.winfo_rootx() > form.ocr_language_box.winfo_rootx()
    assert form.save_areas_checkbox.winfo_rootx() + form.save_areas_checkbox.winfo_width() <= (
        form.save_areas_checkbox.master.winfo_rootx()
        + form.save_areas_checkbox.master.winfo_width()
    )
    assert form.zoom_in_button.winfo_rooty() > (
        form.canvas.winfo_rooty() + form.canvas.winfo_height()
    )
    assert (
        form.zoom_out_button.winfo_rootx() < form.previous_button.winfo_rootx()
        or form.zoom_out_button.winfo_rooty() < form.previous_button.winfo_rooty()
    )
    assert not form.zoom_in_button.cget("text")
    assert form.zoom_in_button.cget("image")
    assert not form.add_area_button.cget("text")
    assert not form.remove_area_button.cget("text")
    assert form.add_area_button.cget("image") and form.remove_area_button.cget("image")


def area_point(form, x, y):
    left, top, width, height = form.area_editor.image_box
    return {
        "x": round(left + x * width - form.canvas.canvasx(0)),
        "y": round(top + y * height - form.canvas.canvasy(0)),
    }


def draw_area(form, start=(0.1, 0.2), end=(0.6, 0.7)):
    if form.area_editor.mode != "add":
        form.add_area_button.invoke()
    form.canvas.event_generate("<ButtonPress-1>", **area_point(form, *start))
    form.canvas.event_generate("<B1-Motion>", **area_point(form, *end))
    form.canvas.event_generate("<ButtonRelease-1>", **area_point(form, *end))


def test_draw_zoom_remove_and_restore_areas(form, image_path, tmp_path):
    from formextractor.areas import source_key
    from formextractor.settings import load_settings

    form.root.deiconify()
    form.root.update()
    form.load_document(image_path)
    draw_area(form)
    assert form.canvas.cget("cursor") == "crosshair"
    assert len(form.area_editor.areas) == 1
    first = form.area_editor.areas[0]
    assert (first.x1, first.y1, first.x2, first.y2) == pytest.approx((0.1, 0.2, 0.6, 0.7), abs=0.01)
    assert first.number == first.page == 1
    assert form.recognize_button.instate(["!disabled"])
    assert len(form.canvas.find_withtag("areas")) == 2
    form.zoom(3)
    assert form.area_editor.areas[0] == first
    rectangle, label = form.canvas.find_withtag("areas")
    left, top, width, height = form.area_editor.image_box
    assert form.canvas.coords(rectangle) == pytest.approx(
        [
            left + first.x1 * width,
            top + first.y1 * height,
            left + first.x2 * width,
            top + first.y2 * height,
        ]
    )
    assert form.canvas.itemcget(label, "text") == "1"
    draw_area(form, (0.7, 0.2), (0.9, 0.6))
    assert [area.number for area in form.area_editor.areas] == [1, 2]
    form.remove_area_button.invoke()
    form.canvas.event_generate("<Motion>", **area_point(form, 0.3, 0.4))
    assert form.canvas.cget("cursor") == "arrow"
    assert form.canvas.find_withtag("remove_pointer")
    form.canvas.event_generate("<ButtonPress-1>", **area_point(form, 0.3, 0.4))
    assert [area.number for area in form.area_editor.areas] == [2]
    stored = load_settings(form.preferences_path)["files"][source_key(image_path)]["areas"]
    assert [area["number"] for area in stored] == [2]
    other = tmp_path / "other.png"
    other.write_bytes(image_path.read_bytes())
    form.load_document(other)
    assert not form.area_editor.areas
    form.load_document(image_path)
    assert [area.number for area in form.area_editor.areas] == [2]
    assert len(form.canvas.find_withtag("areas")) == 2


def test_areas_are_page_specific_and_drag_is_clamped(form, tmp_path):
    source = tmp_path / "pages.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=300)
        document.new_page(width=200, height=300)
        document.save(source)
    form.root.deiconify()
    form.root.update()
    form.load_document(source)
    draw_area(form, (0.4, 0.4), (0.1, 0.1))
    assert form.area_editor.areas[0].x1 < form.area_editor.areas[0].x2
    form.turn_page(1)
    assert not form.canvas.find_withtag("areas")
    draw_area(form, (0.5, 0.5), (1.2, 1.2))
    assert form.area_editor.areas[-1].page == 2
    assert form.area_editor.areas[-1].x2 == form.area_editor.areas[-1].y2 == 1.0
    assert len(form.canvas.find_withtag("areas")) == 2
    form.turn_page(-1)
    assert len(form.canvas.find_withtag("areas")) == 2


def test_window_language_mask_and_templates_survive_reopen(desktop, tmp_path, image_path):
    preferences = tmp_path / "persistent.json"
    root = tk.Toplevel(desktop)
    window = MainForm(root, preferences_path=preferences)
    root.geometry("950x680+80+90")
    root.update()
    window.load_document(image_path)
    draw_area(window)
    window.ocr_language.set(
        next(label for label, code in window.ocr_choices.items() if code == "uk")
    )
    window.file_mask.set("scan*.png")
    window.save_areas_checkbox.invoke()
    window.zoom(2)
    zoom = window.zoom_level
    window.panes.sashpos(0, 400)
    root.update()
    geometry = root.geometry()
    fraction = window.panes.sashpos(0) / window.panes.winfo_width()
    window.close()
    root = tk.Toplevel(desktop)
    reopened = MainForm(root, preferences_path=preferences)
    root.update()
    try:
        assert root.geometry() == geometry
        assert reopened.ocr_choices[reopened.ocr_language.get()] == "uk"
        assert reopened.file_mask.get() == "scan*.png"
        assert reopened.save_areas.get() is True
        assert reopened.zoom_level == zoom
        assert reopened.panes.sashpos(0) / reopened.panes.winfo_width() == pytest.approx(
            fraction, abs=0.005
        )
        assert reopened.source == image_path
        assert reopened.zoom_level == zoom
        assert len(reopened.area_editor.areas) == 1
        assert len(reopened.canvas.find_withtag("areas")) == 2
    finally:
        reopened._shutdown()


@pytest.mark.parametrize("remove_areas", [False, True])
def test_close_saves_latest_area_coordinates_and_reopens_last_page(desktop, tmp_path, remove_areas):
    from formextractor.areas import source_key
    from formextractor.settings import load_settings

    source = tmp_path / "pages.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=300)
        document.new_page(width=300, height=200)
        document.save(source)
    preferences = tmp_path / "settings.json"
    window = MainForm(tk.Toplevel(desktop), preferences_path=preferences)
    window.load_document(source)
    window.area_editor.areas = [
        RecognitionArea(number=1, page=1, x1=0.1, y1=0.2, x2=0.5, y2=0.7),
        RecognitionArea(number=3, page=2, x1=0.25, y1=0.3, x2=0.8, y2=0.9),
    ]
    window._areas_changed()
    # Closing must flush the latest state even without an autosave notification.
    window.area_editor.areas = (
        []
        if remove_areas
        else [
            window.area_editor.areas[0],
            RecognitionArea(number=3, page=2, x1=0.2, y1=0.1, x2=0.6, y2=0.75),
        ]
    )
    expected = list(window.area_editor.areas)
    window.turn_page(1)
    window.close()
    stored = load_settings(preferences)
    assert stored["files"][source_key(source)]["areas"] == [area.model_dump() for area in expected]
    assert stored["last_source"] == str(source.resolve())
    assert stored["last_page"] == 2
    reopened = MainForm(tk.Toplevel(desktop), preferences_path=preferences)
    try:
        reopened.root.update()
        assert reopened.source == source
        assert reopened.page_number == reopened.area_editor.page == 2
        assert reopened.area_editor.areas == expected
        assert len(reopened.canvas.find_withtag("areas")) == (0 if remove_areas else 2)
    finally:
        reopened._shutdown()


@pytest.mark.parametrize("saved_page", [0, 99, "2", None, True])
def test_invalid_saved_page_restores_first_page(desktop, tmp_path, image_path, saved_page):
    from formextractor.settings import update_settings

    preferences = tmp_path / "settings.json"
    update_settings({"last_source": str(image_path), "last_page": saved_page}, preferences)
    reopened = MainForm(tk.Toplevel(desktop), preferences_path=preferences)
    try:
        assert reopened.source == image_path
        assert reopened.page_number == 1
    finally:
        reopened._shutdown()


@pytest.mark.parametrize("source_kind", ["missing", "broken", "invalid"])
def test_unavailable_last_document_keeps_saved_areas(desktop, tmp_path, monkeypatch, source_kind):
    from formextractor.areas import source_key
    from formextractor.settings import load_settings, update_settings

    source = tmp_path / "scan.png"
    if source_kind == "broken":
        source.write_bytes(b"not an image")
    preferences = tmp_path / "settings.json"
    files = {
        source_key(source): {
            "areas": [
                RecognitionArea(number=1, page=1, x1=0.1, y1=0.2, x2=0.5, y2=0.7).model_dump()
            ]
        }
    }
    update_settings(
        {"last_source": [] if source_kind == "invalid" else str(source), "files": files},
        preferences,
    )
    errors = []
    monkeypatch.setattr(
        "formextractor.mainform.messagebox.showerror", lambda *a, **k: errors.append(a)
    )
    reopened = MainForm(tk.Toplevel(desktop), preferences_path=preferences)
    try:
        assert reopened.source is None
        assert reopened.preview is None
        assert not reopened.area_editor.areas
        assert not errors
        reopened.close()
        assert load_settings(preferences)["files"] == files
    finally:
        if reopened.root.winfo_exists():
            reopened._shutdown()


def test_masked_batch_request_displays_and_saves_tabular_results(
    form, image_path, tmp_path, monkeypatch
):
    from formextractor.areas import BatchResult, BatchRow

    form.load_document(image_path)
    add_test_area(form)
    (tmp_path / "scan2.png").write_bytes(image_path.read_bytes())
    (tmp_path / "ignore.png").write_bytes(image_path.read_bytes())
    form.file_mask.set("scan*.png")
    form.save_areas.set(True)
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    assert [Path(path).name for path in job.options["sources"]] == ["scan.png", "scan2.png"]
    assert job.options["mask"] == "scan*.png"
    assert job.options["save_areas"] is True
    assert form.save_areas_checkbox.instate(["disabled"])
    assert job.options["areas"] == [form.area_editor.areas[0].model_dump()]
    assert form.mask_entry.instate(["disabled"])
    assert form.add_area_button.instate(["disabled"])
    result = BatchResult(
        source=str(image_path),
        language="en",
        file_mask="scan*.png",
        areas=list(form.area_editor.areas),
        columns=[1],
        rows=[BatchRow(file=path, values=["value"]) for path in job.options["sources"]],
    )
    job.message = ("batch_result", result.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    header = form.text_editor.get("1.0", "1.end")
    assert [cell.strip() for cell in header.split("|")[1:-1]] == ["File", "Area 1"]
    assert "| scan2.png | value" in form.text_editor.get("1.0", "end")
    assert json.loads(form.json_editor.get("1.0", "end"))["columns"] == [1]
    assert "2 file(s)" in form.status.get()
    assert form.notebook.select() == str(form.text_editor.master)
    destination = tmp_path / "batch.json"
    monkeypatch.setattr(
        "formextractor.mainform.filedialog.asksaveasfilename", lambda **kwargs: str(destination)
    )
    form.save()
    assert BatchResult.model_validate_json(destination.read_text(encoding="utf-8")) == result
    form.json_editor.delete("1.0", "end")
    form.json_editor.insert("1.0", '{"columns": []}')
    form.save()
    assert BatchResult.model_validate_json(destination.read_text(encoding="utf-8")) == result
    assert form.test_errors[-1][0] == "Cannot save result"


def test_no_areas_or_no_mask_matches_cannot_start_recognition(form, image_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "formextractor.mainform.messagebox.showwarning", lambda *a, **k: warnings.append(a)
    )
    form.load_document(image_path)
    form.job_factory = FakeJob
    form.recognize()
    assert warnings and form.job is None
    add_test_area(form)
    form.file_mask.set("does-not-exist*.pdf")
    form.recognize()
    assert form.job is None
    assert form.test_errors[-1][1] == "No supported files match the file mask."


def test_progress_updates_current_file_and_stop_cancels_polling(form, image_path, result):
    class ProgressJob(FakeJob):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.notices = []

        def poll(self):
            return self.notices.pop(0) if self.notices else None

    recognize_result(form, image_path, result)
    previous_text = form.text_editor.get("1.0", "end-1c")
    assert form.stop_button.instate(["disabled"])
    form.job_factory = ProgressJob
    form.recognize()
    job = form.job
    assert image_path.name in form.status.get()
    assert form.stop_button.instate(["!disabled"])
    job.notices = [("progress", "folder/first.pdf"), ("progress", "folder/current.pdf")]
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert form.status.get() == "Recognizing: current.pdf"
    assert form.job is job and not job.closed
    pending_poll = form.poll_id
    assert pending_poll is not None
    form.stop_button.invoke()
    assert job.closed
    assert form.job is None and form.poll_id is None
    assert pending_poll not in form.root.tk.call("after", "info")
    assert form.stop_button.instate(["disabled"])
    assert form.recognize_button.instate(["!disabled"])
    assert form.status.get() == "Recognition stopped."
    assert form.text_editor.get("1.0", "end-1c") == previous_text
    # No stale queued notice from the canceled job can replace the stopped status.
    form._poll_job()
    assert form.status.get() == "Recognition stopped."
    form.recognize()
    assert form.job is not job and form.job.started
    form.stop_button.invoke()


def test_splitter_mask_and_zoom_are_saved_without_closing(form, image_path):
    from formextractor.settings import load_settings

    form.root.deiconify()
    form.root.update()
    form.load_document(image_path)
    form.panes.sashpos(0, 320)
    form.panes.event_generate("<ButtonRelease-1>")
    form.file_mask.set("invoice*.png")
    form.zoom(2)
    form.root.after(450, form.root.quit)
    form.root.mainloop()
    stored = load_settings(form.preferences_path)
    assert stored["file_mask"] == "invoice*.png"
    assert stored["zoom_level"] == form.zoom_level
    assert stored["split_fraction"] == pytest.approx(320 / form.panes.winfo_width())
    assert form.preferences_id is None


def test_area_results_display_before_completion_and_survive_stop(form, image_path):
    from formextractor.areas import BatchResult, BatchRow

    form.load_document(image_path)
    add_test_area(form)
    form.area_editor.areas.append(RecognitionArea(number=2, page=1, x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    form.job_factory = FakeJob
    form.recognize()
    job = form.job
    row = BatchRow(file=str(image_path), values=["First area", None])
    job.message = ("row_result", row.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert form.job is job and not job.closed
    assert "First area" in form.text_editor.get("1.0", "end-1c")
    partial = BatchResult.model_validate_json(form.json_editor.get("1.0", "end-1c"))
    assert partial.rows == [row]
    assert form.text_editor.cget("state") == "disabled"
    assert form.save_button.instate(["disabled"])

    row.values[1] = "Second area"
    job.message = ("row_result", row.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert len(form.result.rows) == 1
    assert form.result.rows[0].values == ["First area", "Second area"]
    form.stop_recognition()
    assert "Second area" in form.text_editor.get("1.0", "end-1c")
    assert form.save_button.instate(["!disabled"])

    # A new run replaces partial rows instead of retaining a prior run's values.
    form.recognize()
    row.values = ["New run", None]
    form.job.message = ("row_result", row.model_dump_json())
    form.root.after_cancel(form.poll_id)
    form._poll_job()
    assert form.result.rows == [row]
    assert "Second area" not in form.text_editor.get("1.0", "end-1c")
