"""Run recognition outside the UI process; only serialized results cross back."""

import multiprocessing
from queue import Empty

from .paddle import PaddleLayout
from .pipeline import extract_document


def recognize_worker(source: str, messages, lang: str = "en") -> None:
    try:
        result = extract_document(source, layout=PaddleLayout(lang=lang))
        messages.put(("result", result.model_dump_json()))
    except Exception as exc:
        messages.put(("error", str(exc)))


class RecognitionJob:
    def __init__(self, source: str, *, lang: str = "en") -> None:
        context = multiprocessing.get_context("spawn")
        self.messages = context.Queue()
        self.process = context.Process(
            target=recognize_worker, args=(source, self.messages, lang), daemon=True
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

    def close(self) -> None:
        if self.process.pid is not None:
            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            self.process.close()
        self.messages.close()
