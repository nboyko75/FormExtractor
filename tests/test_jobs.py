import multiprocessing
import time
from queue import Queue
from types import SimpleNamespace

from formextractor.areas import BatchResult, BatchRow, RecognitionArea
from formextractor.jobs import RecognitionJob, recognize_worker


def test_spawned_worker_returns_input_error_and_closes(tmp_path):
    job = RecognitionJob(str(tmp_path / "missing.pdf"))
    try:
        job.start()
        deadline = time.monotonic() + 15
        message = None
        while message is None and time.monotonic() < deadline:
            message = job.poll()
            time.sleep(0.02)
        assert message is not None
        assert message[0] == "error"
        assert "Input file does not exist" in message[1]
    finally:
        job.close()


def test_worker_applies_recognition_language(monkeypatch):
    calls = []

    def extract(source, *, layout):
        calls.append((source, layout.lang))
        return SimpleNamespace(model_dump_json=lambda: '{"pages": []}')

    monkeypatch.setattr("formextractor.jobs.extract_document", extract)
    messages = Queue()
    recognize_worker("scan.pdf", messages, "uk")
    assert calls == [("scan.pdf", "uk")]
    assert messages.get_nowait() == ("result", '{"pages": []}')


def test_job_passes_selected_language_to_spawned_process(monkeypatch):
    messages = Queue()
    processes = []

    def process(**kwargs):
        processes.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "formextractor.jobs.multiprocessing.get_context",
        lambda method: SimpleNamespace(Queue=lambda: messages, Process=process),
    )
    RecognitionJob("scan.pdf", lang="de", save_areas=True)
    assert processes[0]["target"] is recognize_worker
    assert processes[0]["args"] == ("scan.pdf", messages, "de")
    assert processes[0]["kwargs"]["save_areas"] is True


def test_spawned_batch_worker_preserves_area_template_and_reports_file_errors(tmp_path):
    filename = str(tmp_path / "missing.pdf")
    area = RecognitionArea(number=2, page=1, x1=0.1, y1=0.2, x2=0.5, y2=0.8)
    job = RecognitionJob(
        filename, lang="uk", sources=[filename], areas=[area.model_dump()], mask="*.pdf"
    )
    try:
        job.start()
        deadline = time.monotonic() + 15
        message = None
        progress = []
        while message is None and time.monotonic() < deadline:
            message = job.poll()
            if message is not None and message[0] in {"progress", "row_result"}:
                if message[0] == "progress":
                    progress.append(message[1])
                message = None
            time.sleep(0.02)
        assert message is not None and message[0] == "batch_result"
        assert progress == [filename]
        result = BatchResult.model_validate_json(message[1])
        assert result.language == "uk"
        assert result.areas == [area]
        assert result.columns == [2]
        assert result.rows[0].file == filename
        assert result.rows[0].values == [None]
        assert "Input file does not exist" in result.rows[0].error
    finally:
        job.close()


def test_stop_terminates_before_waiting_and_is_idempotent(monkeypatch):
    calls = []
    process = SimpleNamespace(pid=123, alive=True)

    def terminate():
        calls.append("terminate")
        process.alive = False

    process.is_alive = lambda: process.alive
    process.terminate = terminate
    process.join = lambda timeout: calls.append("join")
    process.close = lambda: calls.append("close process")
    context = SimpleNamespace(
        Queue=lambda: SimpleNamespace(close=lambda: calls.append("close queue")),
        Process=lambda **kwargs: process,
    )
    monkeypatch.setattr("formextractor.jobs.multiprocessing.get_context", lambda method: context)
    job = RecognitionJob("scan.pdf")
    job.stop()
    assert calls == ["terminate", "join", "close process", "close queue"]
    job.stop()
    job.close()
    assert calls == ["terminate", "join", "close process", "close queue"]


def test_stop_ends_a_running_worker_without_waiting_for_its_work():
    job = RecognitionJob("unused.pdf")
    job.process.close()
    job.process = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(30,), daemon=True
    )
    try:
        job.start()
        pid = job.process.pid
        started = time.monotonic()
        job.stop()
        assert time.monotonic() - started < 3
        assert all(child.pid != pid for child in multiprocessing.active_children())
        assert job._closed
    finally:
        job.close()


def test_worker_streams_partial_rows_before_batch_completion(monkeypatch):
    area = RecognitionArea(number=1, page=1, x1=0, y1=0, x2=1, y2=1)
    row = BatchRow(file="scan.png", values=["Recognized area"])

    def recognize(source, sources, areas, mask, *, language, save_areas, on_file, on_row):
        assert save_areas is True
        on_file(sources[0])
        on_row(row)
        return BatchResult(
            source=source, language=language, file_mask=mask, areas=areas, columns=[1], rows=[row]
        )

    monkeypatch.setattr("formextractor.jobs.recognize_batch", recognize)
    messages = Queue()
    recognize_worker(
        "scan.png", messages, areas=[area.model_dump()], sources=["scan.png"], save_areas=True
    )
    assert messages.get_nowait() == ("progress", "scan.png")
    assert messages.get_nowait() == ("row_result", row.model_dump_json())
    assert messages.get_nowait()[0] == "batch_result"
    assert messages.empty()
