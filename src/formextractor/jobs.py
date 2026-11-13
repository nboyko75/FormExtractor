"""Run recognition outside the UI process; only serialized results cross back."""

import multiprocessing
from queue import Empty

from .areas import RecognitionArea, recognize_batch
from .paddle import PaddleLayout
from .pipeline import extract_document


def recognize_worker(
    source: str, messages, lang: str = "en", *, areas=None, sources=None, mask="*"
) -> None:
    try:
        if areas is not None:
            result = recognize_batch(
                source,
                sources or [],
                [RecognitionArea.model_validate(area) for area in areas],
                mask,
                language=lang,
                on_file=lambda filename: messages.put(("progress", filename)),
            )
            messages.put(("batch_result", result.model_dump_json()))
            return
        result = extract_document(source, layout=PaddleLayout(lang=lang))
        messages.put(("result", result.model_dump_json()))
    except Exception as exc:
        messages.put(("error", str(exc)))


class RecognitionJob:
    def __init__(
        self, source: str, *, lang: str = "en", areas=None, sources=None, mask="*"
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.messages = context.Queue()
        self._closed = False
        self.process = context.Process(
            target=recognize_worker,
            args=(source, self.messages, lang),
            kwargs={"areas": areas, "sources": sources, "mask": mask},
            daemon=True,
        )

    def start(self) -> None:
        self.process.start()

    def poll(self) -> tuple[str, str] | None:
        try:
            return self.messages.get_nowait()
        except Empty:
            if self.process.exitcode is not None:
                return (
                    "error",
                    f"Recognition stopped unexpectedly (exit {self.process.exitcode}).",
                )
            return None

    def stop(self) -> None:
        """Terminate native OCR immediately instead of waiting for a page to finish."""
        if not self._closed and self.process.pid is not None and self.process.is_alive():
            self.process.terminate()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self.process.pid is not None:
            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
        self.messages.close()
        self._closed = True
