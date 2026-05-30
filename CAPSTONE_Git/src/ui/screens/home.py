import tkinter as tk
from tkinter import ttk


H1 = ("Segoe UI", 18, "bold")
H2 = ("Segoe UI", 11)


class HomeScreen(tk.Frame):
    """Main home navigation screen."""

    def __init__(self, master, router):
        super().__init__(master, bg="#f5f5f5")
        self.router = router
        self._build()

    def _build(self):
        header = tk.Frame(self, bg="#1a1a2e", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="NBA Team Analytics Dashboard",
            font=H1,
            bg="#1a1a2e",
            fg="white"
        ).pack(side="left", padx=20, pady=20)

        content = tk.Frame(self, bg="#f5f5f5")
        content.pack(expand=True, fill="both", padx=40, pady=40)

        tk.Label(
            content,
            text="Analyze NBA team performance and predict season outcomes",
            font=H2,
            bg="#f5f5f5",
            fg="#333"
        ).pack(anchor="w", pady=(0, 30))

        btn_frame = tk.Frame(content, bg="#f5f5f5")
        btn_frame.pack(anchor="w")

        style = ttk.Style()
        style.configure("Nav.TButton", font=("Segoe UI", 11), padding=(20, 12))

        ttk.Button(
            btn_frame,
            text="Season Outcome Predictor",
            style="Nav.TButton",
            command=lambda: self.router.go("predict")
        ).pack(side="left", padx=(0, 15))

        ttk.Button(
            btn_frame,
            text="League Trends Analysis",
            style="Nav.TButton",
            command=lambda: self.router.go("analysis")
        ).pack(side="left", padx=(0, 15))

        tk.Label(
            content,
            text="WGU Computer Science Capstone Project\nCreated by S. Curtis\n Jan 2026",
            font=("Segoe UI", 9),
            bg="#f5f5f5",
            fg="#888"
        ).pack(side="bottom", anchor="center", pady=(40, 0))