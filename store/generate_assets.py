"""Generate the FormExtractor document/field icons used by the MSIX manifest."""

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    assets = Path(__file__).resolve().parent / "Assets"
    assets.mkdir(exist_ok=True)
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((160, 64, 864, 960), radius=90, fill="#155E75")
    draw.rounded_rectangle((240, 152, 784, 872), radius=32, fill="white")
    draw.rounded_rectangle((304, 232, 720, 280), radius=16, fill="#155E75")
    for y in (368, 520, 672):
        draw.rounded_rectangle((304, y, 376, y + 72), radius=12, fill="#14B8A6")
        draw.rounded_rectangle((424, y + 16, 720, y + 56), radius=12, fill="#64748B")
    sizes = {"StoreLogo.png": 50, "Logo44.png": 44, "Logo150.png": 150, "Logo512.png": 512}
    sizes.update({f"Logo44.targetsize-{size}_altform-unplated.png": size for size in (16, 32, 48, 256)})
    for name, size in sizes.items():
        image.resize((size, size), Image.Resampling.LANCZOS).save(assets / name)


if __name__ == "__main__":
    main()
