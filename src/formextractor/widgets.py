"""Small desktop controls shared by the main form."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk


def zoom_icon(owner: tk.Misc, *, plus: bool, disabled: bool = False) -> ImageTk.PhotoImage:
    """Draw a magnifying glass with a plus or minus, without font-dependent glyphs."""
    size = max(20, round(20 * owner.winfo_fpixels("1i") / 96))
    image = Image.new("RGBA", (80, 80))
    draw = ImageDraw.Draw(image)
    color = "#9ca3af" if disabled else "#263648"
    draw.line((51, 51, 72, 72), fill=color, width=9)
    draw.ellipse((7, 7, 58, 58), outline=color, width=6)
    draw.line((20, 33, 45, 33), fill=color, width=5)
    if plus:
        draw.line((33, 20, 33, 45), fill=color, width=5)
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    result = ImageTk.PhotoImage(resized, master=owner)
    resized.close()
    image.close()
    return result


class ToolTip:
    def __init__(self, widget: ttk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.pending = None
        self.window = None
        for event in ("<Enter>", "<FocusIn>"):
            widget.bind(event, self.schedule, add=True)
        for event in ("<Leave>", "<FocusOut>", "<ButtonPress>", "<Destroy>"):
            widget.bind(event, self.hide, add=True)

    def schedule(self, event=None) -> None:
        self.hide()
        self.pending = self.widget.after(500, self.show)

    def show(self) -> None:
        self.pending = None
        self.window = tk.Toplevel(self.widget)
        self.window.overrideredirect(True)
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window.geometry(f"+{x}+{y}")
        ttk.Label(self.window, text=self.text, padding=6, relief="solid").pack()

    def hide(self, event=None) -> None:
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
