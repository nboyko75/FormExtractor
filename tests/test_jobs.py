import time
from queue import Queue
from types import SimpleNamespace

from formextractor.areas import BatchResult, RecognitionArea
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
    RecognitionJob("scan.pdf", lang="de")
    assert processes[0]["target"] is recognize_worker
    assert processes[0]["args"] == ("scan.pdf", messages, "de")


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
        while message is None and time.monotonic() < deadline:
            message = job.poll()
            time.sleep(0.02)
        assert message is not None and message[0] == "batch_result"
        result = BatchResult.model_validate_json(message[1])
        assert result.language == "uk"
        assert result.areas == [area]
        assert result.columns == [2]
        assert result.rows[0].file == filename
        assert result.rows[0].values == [None]
        assert "Input file does not exist" in result.rows[0].error
    finally:
        job.close()
