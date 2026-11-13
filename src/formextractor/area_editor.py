"""Preview interactions for drawing and deleting page-relative recognition areas."""

from .areas import RecognitionArea


class AreaEditor:
    def __init__(self, canvas, changed) -> None:
        self.canvas = canvas
        self.changed = changed
        self.areas = []
        self.page = 1
        self.mode = None
        self.enabled = True
        self.image_box = None
        self.start = None
        self.end = None
        canvas.bind("<ButtonPress-1>", self.press)
        canvas.bind("<B1-Motion>", self.drag)
        canvas.bind("<ButtonRelease-1>", self.release)
        canvas.bind("<Motion>", self.pointer)
        canvas.bind("<Leave>", lambda event: canvas.delete("remove_pointer"))

    def set_mode(self, mode) -> None:
        self.mode = mode if self.enabled else None
        self.start = self.end = None
        self.canvas.delete("remove_pointer")
        self.canvas.configure(cursor="crosshair" if self.mode == "add" else "arrow")
        self.render()

    def position(self, event, *, clamp=False):
        if self.image_box is None:
            return None
        left, top, width, height = self.image_box
        x = (self.canvas.canvasx(event.x) - left) / width
        y = (self.canvas.canvasy(event.y) - top) / height
        if not clamp and not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def pointer(self, event) -> None:
        self.canvas.delete("remove_pointer")
        if self.mode == "remove" and self.enabled:
            # Keep the native pointer and attach a minus badge to it. This also
            # works on platforms without a native "pointer-minus" cursor.
            x, y = self.canvas.canvasx(event.x) + 12, self.canvas.canvasy(event.y) + 16
            self.canvas.create_rectangle(
                x - 6,
                y - 6,
                x + 6,
                y + 6,
                fill="white",
                outline="#d00000",
                tags="remove_pointer",
            )
            self.canvas.create_text(
                x,
                y,
                text="−",
                fill="#d00000",
                font=("Segoe UI", 10, "bold"),
                tags="remove_pointer",
            )

    def press(self, event) -> None:
        point = self.position(event)
        if self.mode == "add" and self.enabled and point is not None:
            self.start = self.end = point
            self.render()
        elif self.mode == "remove" and self.enabled and point is not None:
            x, y = point
            for area in reversed(self.areas):
                if area.page == self.page and area.x1 <= x <= area.x2 and area.y1 <= y <= area.y2:
                    self.areas.remove(area)
                    self.render()
                    self.changed()
                    break
        elif self.mode is None:
            self.canvas.scan_mark(event.x, event.y)

    def drag(self, event) -> None:
        if self.start is not None:
            self.end = self.position(event, clamp=True)
            self.render()
        elif self.mode is None:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.pointer(event)

    def release(self, event) -> None:
        if self.start is None:
            return
        self.end = self.position(event, clamp=True)
        x1, x2 = sorted((self.start[0], self.end[0]))
        y1, y2 = sorted((self.start[1], self.end[1]))
        self.start = self.end = None
        if (x2 - x1) * self.image_box[2] >= 3 and (y2 - y1) * self.image_box[3] >= 3:
            self.areas.append(
                RecognitionArea(
                    number=max((area.number for area in self.areas), default=0) + 1,
                    page=self.page,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
            self.changed()
        self.render()

    def rectangle(self, number, x1, y1, x2, y2, *, draft=False) -> None:
        left, top, width, height = self.image_box
        box = left + x1 * width, top + y1 * height, left + x2 * width, top + y2 * height
        self.canvas.create_rectangle(
            *box,
            outline="#e00000",
            width=2,
            dash=(4, 2) if draft else (),
            tags="areas",
        )
        self.canvas.create_text(
            box[0],
            box[1] - 3,
            text=str(number),
            fill="#d00000",
            anchor="sw",
            font=("Segoe UI", 10, "bold"),
            tags="areas",
        )

    def render(self) -> None:
        self.canvas.delete("areas")
        if self.image_box is None:
            return
        for area in self.areas:
            if area.page == self.page:
                self.rectangle(area.number, area.x1, area.y1, area.x2, area.y2)
        if self.start is not None and self.end is not None:
            x1, x2 = sorted((self.start[0], self.end[0]))
            y1, y2 = sorted((self.start[1], self.end[1]))
            self.rectangle(
                max((area.number for area in self.areas), default=0) + 1,
                x1,
                y1,
                x2,
                y2,
                draft=True,
            )
