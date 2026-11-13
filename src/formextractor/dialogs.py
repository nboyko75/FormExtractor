"""Settings and About windows for the desktop application."""

import sys
import tkinter as tk
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tkinter import ttk
from xml.etree import ElementTree

from .settings import LANGUAGES

WEBSITE = "https://nboyko75.github.io/NBWebsite/"


def app_version() -> str:
    # MSIX can override the version after the executable has been built.
    if getattr(sys, "frozen", False):
        manifest = Path(sys.executable).parent / "AppxManifest.xml"
        try:
            identity = ElementTree.parse(manifest).find(
                "{http://schemas.microsoft.com/appx/manifest/foundation/windows10}Identity"
            )
            if identity is not None and identity.get("Version"):
                return identity.get("Version")
        except (OSError, ElementTree.ParseError):
            pass
    try:
        return version("formextractor")
    except PackageNotFoundError:
        return "0.1.0"


def prepare_dialog(window: tk.Toplevel, owner: tk.Misc) -> None:
    window.transient(owner)
    window.resizable(False, False)
    window.update_idletasks()
    x = owner.winfo_rootx() + max(0, (owner.winfo_width() - window.winfo_reqwidth()) // 2)
    y = owner.winfo_rooty() + max(0, (owner.winfo_height() - window.winfo_reqheight()) // 2)
    window.geometry(f"+{x}+{y}")
    window.grab_set()


class SettingsForm:
    def __init__(self, owner, tr, on_close) -> None:
        self.window = tk.Toplevel(owner)
        self.window.title(tr("settings").rstrip("."))
        self.on_close = on_close
        self.language = tk.StringVar(master=self.window, value=LANGUAGES[tr.language])
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=16, pady=16)
        main = ttk.Frame(notebook, padding=20)
        notebook.add(main, text=tr("main"))
        ttk.Label(main, text=tr("language")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.language_box = ttk.Combobox(
            main,
            textvariable=self.language,
            values=list(LANGUAGES.values()),
            state="readonly",
            width=34,
        )
        self.language_box.grid(row=1, column=0, sticky="ew")
        buttons = ttk.Frame(self.window, padding=(16, 0, 16, 16))
        buttons.pack(fill="x")
        self.cancel_button = ttk.Button(buttons, text=tr("cancel"), command=self.close)
        self.cancel_button.pack(side="right")
        self.ok_button = ttk.Button(buttons, text=tr("ok"), command=self.apply)
        self.ok_button.pack(side="right", padx=(0, 8))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.close())
        self.window.bind("<Return>", lambda event: self.apply())
        prepare_dialog(self.window, owner)
        self.language_box.focus_set()

    def close(self) -> None:
        self.window.destroy()

    def apply(self) -> None:
        code = next(code for code, name in LANGUAGES.items() if name == self.language.get())
        self.on_close(code)


class AboutForm:
    def __init__(self, owner, tr) -> None:
        self.window = tk.Toplevel(owner)
        self.window.title(tr("about"))
        panel = ttk.Frame(self.window, padding=(24, 16))
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="FormExtractor", font=("Segoe UI", 16, "bold")).pack(pady=(0, 12))
        self.version_label = ttk.Label(panel, text=tr("version", version=app_version()))
        self.version_label.pack(pady=(0, 6))
        ttk.Label(panel, text=tr("description"), wraplength=380, justify="center").pack(pady=6)
        ttk.Label(panel, text="(c) Nick Boiko").pack(pady=3)
        ttk.Label(panel, text="nboyko75@gmail.com").pack(pady=3)
        link = ttk.Label(panel, text=WEBSITE, foreground="#0563c1", cursor="hand2", takefocus=True)
        link.pack(pady=(3, 16))
        link.bind("<Button-1>", lambda event: webbrowser.open(WEBSITE))
        link.bind("<Return>", lambda event: webbrowser.open(WEBSITE))
        self.close_button = ttk.Button(panel, text=tr("close"), command=self.window.destroy)
        self.close_button.pack()
        self.window.bind("<Escape>", lambda event: self.window.destroy())
        self.window.bind("<Return>", lambda event: self.window.destroy())
        prepare_dialog(self.window, owner)
        self.close_button.focus_set()
