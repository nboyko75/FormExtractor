"""Create a synthetic scanned form in PDF and PNG for a real-model smoke test.

Run after installing .[dev]: python examples/create_sample.py
"""

from pathlib import Path

import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def main() -> None:
    output = Path("output/sample")
    output.mkdir(parents=True, exist_ok=True)
    original = output / "vehicle_form_original.pdf"
    page = canvas.Canvas(str(original), pagesize=A4)
    page.setFont("Helvetica-Bold", 20)
    page.drawString(45, 790, "Vehicle Inspection Report")
    page.setFont("Helvetica", 13)
    for y, label, value in [
        (735, "1. Vehicle model", "Skoda Yeti"),
        (695, "2. Year", "2011"),
        (655, "3. Mileage", "102000 km"),
        (615, "4. Inspection ID", "00123"),
    ]:
        page.drawString(45, y, label)
        page.drawString(270, y, value)
        page.line(265, y - 5, 550, y - 5)
    page.setFont("Helvetica-Bold", 14)
    page.drawString(45, 555, "Technical condition")
    for y in (530, 495, 460, 425):
        page.line(45, y, 550, y)
    for x in (45, 260, 550):
        page.line(x, 425, x, 530)
    page.setFont("Helvetica", 12)
    for y, left, right in [
        (508, "Component", "Condition"),
        (473, "Engine", "Good"),
        (438, "Transmission", "Inspection required"),
    ]:
        page.drawString(55, y, left)
        page.drawString(270, y, right)
    page.save()
    with pymupdf.open(original) as document:
        document[0].get_pixmap(dpi=150).save(output / "vehicle_form.png")
    scanned = canvas.Canvas(str(output / "vehicle_form.pdf"), pagesize=A4)
    scanned.drawImage(str(output / "vehicle_form.png"), 0, 0, width=A4[0], height=A4[1])
    scanned.save()
    print(f"Created {output / 'vehicle_form.pdf'} and vehicle_form.png")


if __name__ == "__main__":
    main()
