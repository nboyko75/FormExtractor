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


def circle_icon(owner: tk.Misc, *, plus: bool, disabled: bool = False) -> ImageTk.PhotoImage:
    """Draw the area-action circle and plus/minus at the current display scale."""
    size = max(20, round(20 * owner.winfo_fpixels("1i") / 96))
    image = Image.new("RGBA", (80, 80))
    draw = ImageDraw.Draw(image)
    color = "#9ca3af" if disabled else "#263648"
    draw.ellipse((7, 7, 73, 73), outline=color, width=6)
    draw.line((24, 40, 56, 40), fill=color, width=6)
    if plus:
        draw.line((40, 24, 40, 56), fill=color, width=6)
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    result = ImageTk.PhotoImage(resized, master=owner)
    resized.close()
    image.close()
    return result


def arrow_icon(owner: tk.Misc, *, right: bool, disabled: bool = False) -> ImageTk.PhotoImage:
    """Draw a page-navigation arrow at the current display scale."""
    size = max(20, round(20 * owner.winfo_fpixels("1i") / 96))
    image = Image.new("RGBA", (80, 80))
    draw = ImageDraw.Draw(image)
    color = "#9ca3af" if disabled else "#263648"
    points = (
        (10, 36), (51, 36), (35, 20), (41, 14), (67, 40),
        (41, 66), (35, 60), (51, 44), (10, 44),
    )
    if not right:
        points = tuple((80 - x, y) for x, y in points)
    draw.polygon(points, fill=color)
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    result = ImageTk.PhotoImage(resized, master=owner)
    resized.close()
    image.close()
    return result


def exit_icon(owner: tk.Misc) -> ImageTk.PhotoImage:
    """Draw a door and red exit arrow like DocExplorer's wx.ART_QUIT icon."""
    size = max(20, round(20 * owner.winfo_fpixels("1i") / 96))
    image = Image.new("RGBA", (80, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 4, 51, 75), fill="#5c6063", outline="#34383b", width=3)
    draw.rectangle((11, 10, 45, 69), fill="#f0f1f2")
    draw.polygon(((11, 10), (38, 19), (38, 63), (11, 69)), fill="#b8bdc2")
    draw.line((11, 10, 38, 19, 38, 63, 11, 69), fill="#858b91", width=2)
    draw.ellipse((29, 39, 33, 44), fill="#555b61")
    arrow = ((75, 31), (57, 31), (57, 20), (36, 40), (57, 60), (57, 49), (75, 49))
    draw.polygon(arrow, fill="#d62828", outline="#8e1717", width=3)
    draw.line((72, 34, 54, 34, 54, 28, 41, 40), fill="#f47b72", width=2)
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
