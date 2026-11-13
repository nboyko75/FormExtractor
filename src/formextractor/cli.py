import argparse
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .output import to_text, validate_output_path, write_output
from .paddle import PaddleLayout
from .pipeline import extract_document


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Recognize scanned form text with local PaddleOCR PP-StructureV3."
    )
    result.add_argument("input", type=Path, help="PDF, JPG, PNG, TIFF, BMP, or WebP file")
    result.add_argument("--format", choices=["json", "text"], default="json")
    result.add_argument("-o", "--output", type=Path, help="UTF-8 output file; default: stdout")
    result.add_argument("--pages", help="One-based page selection, e.g. 1,3-5")
    result.add_argument("--dpi", type=positive_int, default=200)
    result.add_argument("--max-pixels", type=positive_int, default=4_000_000)
    result.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Additional clockwise rotation after PDF/EXIF orientation",
    )
    result.add_argument("--ocr-device", default="cpu", help="Paddle device, e.g. cpu or gpu:0")
    result.add_argument("--ocr-lang", default="en", help="PaddleOCR language code")
    result.add_argument("--verbose", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.output:
            validate_output_path(args.output, args.input)
        # Keep Python-level model/library diagnostics out of JSON stdout.
        with redirect_stdout(sys.stderr):
            layout = PaddleLayout(device=args.ocr_device, lang=args.ocr_lang)
            result = extract_document(
                args.input,
                pages=args.pages,
                dpi=args.dpi,
                max_pixels=args.max_pixels,
                rotation=args.rotate,
                layout=layout,
            )
        content = (
            result.model_dump_json(indent=2) + "\n" if args.format == "json" else to_text(result)
        )
        if args.output:
            write_output(args.output, content, source=args.input)
            logging.info("Saved %s", args.output)
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(content)
        return 0
    except KeyboardInterrupt:
        print("Extraction cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.error("%s", exc, exc_info=args.verbose)
        return 1
