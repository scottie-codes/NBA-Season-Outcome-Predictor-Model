import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd

from src.models.predictor import get_seasons, get_teams, predict_for
from src.models.data_io import load_master, get_team_row

H1 = ("Segoe UI", 16, "bold")
H2 = ("Segoe UI", 10)

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"


def _load_metrics():
    """Load saved model metrics for display."""
    metrics = {}

    cls_path = PROC / "metrics_classification.csv"
    if cls_path.exists():
        df = pd.read_csv(cls_path)
        if "ACC" in df.columns:
            metrics["acc"] = df["ACC"].iloc[0]
        if "AUC" in df.columns:
            metrics["auc"] = df["AUC"].iloc[0]

    reg_path = PROC / "metrics_regression.csv"
    if reg_path.exists():
        df = pd.read_csv(reg_path)
        if "MAE" in df.columns:
            metrics["mae"] = df["MAE"].iloc[0]
        if "R2" in df.columns:
            metrics["r2"] = df["R2"].iloc[0]

    return metrics


class PredictScreen(tk.Frame):
    """Screen for running predictions on team season outcomes."""

    def __init__(self, master, router):
        super().__init__(master, bg="white")
        self.router = router
        self._teams_df = None
        self._metrics = _load_metrics()
        self._build()

    def _build(self):
        header = tk.Frame(self, bg="#1a1a2e", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Season Outcome Predictor",
            font=H1,
            bg="#1a1a2e",
            fg="white"
        ).pack(side="left", padx=20, pady=15)

        ttk.Button(
            header,
            text="← Back",
            command=lambda: self.router.go("home")
        ).pack(side="right", padx=20, pady=15)

        content = tk.Frame(self, bg="white")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(
            content,
            text="Select a season and team to predict win percentage and playoff chances using Random Forest models.",
            font=H2,
            bg="white",
            fg="#444"
        ).pack(anchor="w", pady=(0, 15))

        controls = tk.Frame(content, bg="white")
        controls.pack(anchor="w", pady=(0, 15))

        tk.Label(controls, text="Season:", bg="white", font=H2).pack(side="left")
        self.cbo_season = ttk.Combobox(controls, width=12, state="readonly", values=get_seasons())
        if self.cbo_season["values"]:
            self.cbo_season.current(len(self.cbo_season["values"]) - 1)
        self.cbo_season.pack(side="left", padx=(8, 20))
        self.cbo_season.bind("<<ComboboxSelected>>", self._load_teams)

        tk.Label(controls, text="Team:", bg="white", font=H2).pack(side="left")
        self.cbo_team = ttk.Combobox(controls, width=28, state="readonly", values=[])
        self.cbo_team.pack(side="left", padx=(8, 20))

        ttk.Button(controls, text="Run Prediction", command=self._on_predict).pack(side="left")

        panels_frame = tk.Frame(content, bg="white")
        panels_frame.pack(fill="both", expand=True, pady=(15, 0))

        # Left panel - Prediction Results
        left_frame = tk.Frame(panels_frame, bg="white")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left_frame, text="Prediction Results", font=("Segoe UI", 11, "bold"), bg="white").pack(anchor="w")

        self.txt_predict = tk.Text(
            left_frame,
            width=50,
            height=24,
            bg="#1a1a2e",
            fg="#e0e0e0",
            font=("Consolas", 10),
            padx=15,
            pady=15,
            cursor="arrow"
        )
        self.txt_predict.pack(fill="both", expand=True, pady=(5, 0))
        self._setup_tags(self.txt_predict)

        # Right panel - Model Performance
        right_frame = tk.Frame(panels_frame, bg="white")
        right_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(right_frame, text="Model Performance", font=("Segoe UI", 11, "bold"), bg="white").pack(anchor="w")

        self.txt_model = tk.Text(
            right_frame,
            width=50,
            height=24,
            bg="#1a1a2e",
            fg="#e0e0e0",
            font=("Consolas", 10),
            padx=15,
            pady=15,
            cursor="arrow"
        )
        self.txt_model.pack(fill="both", expand=True, pady=(5, 0))
        self._setup_tags(self.txt_model)

        for txt in [self.txt_predict, self.txt_model]:
            txt.bind("<Button-3>", lambda e, t=txt: self._show_context_menu(e, t))
            txt.bind("<Control-c>", lambda e, t=txt: self._copy_text(t))

        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Copy", command=lambda: self._copy_text(self._active_txt))
        self.context_menu.add_command(label="Select All", command=lambda: self._select_all(self._active_txt))
        self._active_txt = self.txt_predict

        self._load_teams()

    def _setup_tags(self, txt):
        """Configure text tags for colors."""
        txt.tag_configure("header", foreground="#5dade2", font=("Consolas", 11, "bold"))
        txt.tag_configure("section", foreground="#85c1e9", font=("Consolas", 10, "bold"))
        txt.tag_configure("positive", foreground="#58d68d")
        txt.tag_configure("negative", foreground="#f5b041")
        txt.tag_configure("neutral", foreground="#e0e0e0")
        txt.tag_configure("muted", foreground="#b0b0b0")
        txt.tag_configure("divider", foreground="#666666")
        txt.tag_configure("error_bad", foreground="#e74c3c")
        txt.tag_configure("error_ok", foreground="#f5b041")
        txt.tag_configure("error_good", foreground="#58d68d")
        txt.tag_configure("italic", foreground="#e0e0e0", font=("Consolas", 10, "italic"))

    def _show_context_menu(self, event, txt):
        self._active_txt = txt
        self.context_menu.post(event.x_root, event.y_root)

    def _copy_text(self, txt):
        try:
            selected = txt.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(selected)
        except tk.TclError:
            pass

    def _select_all(self, txt):
        txt.tag_add("sel", "1.0", "end")
        txt.focus_set()

    def _insert(self, txt, text, tag="neutral"):
        txt.insert("end", text, tag)

    def _load_teams(self, *_):
        season = self.cbo_season.get()
        if not season:
            self.cbo_team["values"] = []
            return
        try:
            df = get_teams(season)
            self._teams_df = df.reset_index(drop=True)
            if "TEAM_NAME" in self._teams_df.columns:
                display_vals = self._teams_df["TEAM_NAME"].astype(str).tolist()
            else:
                display_vals = self._teams_df["TEAM_ID"].astype(int).astype(str).tolist()
            self.cbo_team["values"] = display_vals
            if display_vals:
                self.cbo_team.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load teams: {e}")

    def _on_predict(self):
        season = self.cbo_season.get()
        if not season or self._teams_df is None or self._teams_df.empty:
            messagebox.showwarning("Selection Required", "Please select a season and team.")
            return

        idx = self.cbo_team.current()
        if idx < 0:
            messagebox.showwarning("Selection Required", "Please select a team.")
            return

        team_id = int(self._teams_df.iloc[idx]["TEAM_ID"])
        team_name = self._teams_df.iloc[idx].get("TEAM_NAME", "Unknown")

        try:
            result = predict_for(season, team_id)
        except FileNotFoundError as e:
            messagebox.showerror(
                "Models Not Found",
                f"Trained models not found.\n\n{e}\n\nEnsure classifier.pkl and regressor.pkl exist in data/processed/"
            )
            return
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        # Get MADE_PLAYOFFS from data
        made_playoffs = None
        try:
            row = get_team_row(season, team_id)
            if "MADE_PLAYOFFS" in row.index:
                made_playoffs = int(row["MADE_PLAYOFFS"])
        except Exception:
            pass

        mae = self._metrics.get("mae", 0.086)
        r2 = self._metrics.get("r2", 0.33)
        acc = self._metrics.get("acc", 0.57)
        auc = self._metrics.get("auc", 0.65)

        display_name = result.get('team_name') or team_name
        pred_win = result.get("win_pct_pred")
        actual_win = result.get("win_pct_actual")
        prob_top = result.get("prob_top")
        pred_top = result.get("pred_top")

        # Playoff prediction - two formats
        pred_playoffs_label = "MAKE" if pred_top == 1 else "MISS"
        pred_playoffs_yn = "Yes" if pred_top == 1 else "No"
        pred_playoffs_tag = "positive" if pred_top == 1 else "negative"

        # Left panel
        self.txt_predict.delete("1.0", "end")

        self._insert(self.txt_predict, f"{display_name.upper()}\n", "header")
        self._insert(self.txt_predict, f"{season} Season\n", "muted")
        self._insert(self.txt_predict, "=" * 44 + "\n\n", "divider")

        self._insert(self.txt_predict, "PREDICTION SUMMARY\n", "section")
        self._insert(self.txt_predict, "-" * 44 + "\n", "divider")

        self._insert(self.txt_predict, "The ", "neutral")
        self._insert(self.txt_predict, "prediction model", "italic")
        self._insert(self.txt_predict, " estimates this\nteam will win approximately ", "neutral")

        if pred_win is not None:
            win_tag = "positive" if pred_win >= 0.5 else "negative"
            self._insert(self.txt_predict, f"{pred_win:.0%}", win_tag)

        self._insert(self.txt_predict, " of games.\n\n", "neutral")

        if pred_top is not None:
            self._insert(self.txt_predict, "Predicted playoff result: ", "neutral")
            self._insert(self.txt_predict, f"{pred_playoffs_label} PLAYOFFS\n", pred_playoffs_tag)

        self._insert(self.txt_predict, "\n", "neutral")

        self._insert(self.txt_predict, "RESULTS COMPARISON\n", "section")
        self._insert(self.txt_predict, "-" * 44 + "\n", "divider")
        self._insert(self.txt_predict, f"{'':24}{'PRED':>10}{'ACTUAL':>10}\n", "muted")

        self._insert(self.txt_predict, f"{'Win Percentage:':<24}", "neutral")
        if pred_win is not None:
            self._insert(self.txt_predict, f"{pred_win:>9.1%}", "neutral")
        else:
            self._insert(self.txt_predict, f"{'--':>10}", "muted")
        if actual_win is not None:
            self._insert(self.txt_predict, f"{actual_win:>10.1%}\n", "neutral")
        else:
            self._insert(self.txt_predict, f"{'--':>10}\n", "muted")

        self._insert(self.txt_predict, f"{'Made Playoffs:':<24}", "neutral")
        self._insert(self.txt_predict, f"{pred_playoffs_yn:>10}", pred_playoffs_tag)
        if made_playoffs is not None:
            actual_playoff = "Yes" if made_playoffs == 1 else "No"
            tag = "positive" if made_playoffs == 1 else "negative"
            self._insert(self.txt_predict, f"{actual_playoff:>10}\n", tag)
        else:
            self._insert(self.txt_predict, f"{'--':>10}\n", "muted")

        if pred_win is not None and actual_win is not None:
            error = abs(pred_win - actual_win)
            self._insert(self.txt_predict, "\n", "neutral")
            self._insert(self.txt_predict, f"Prediction Error: ", "neutral")
            if error <= 0.05:
                self._insert(self.txt_predict, f"{error:.1%} ", "error_good")
                self._insert(self.txt_predict, "(Excellent)\n", "error_good")
            elif error <= 0.10:
                self._insert(self.txt_predict, f"{error:.1%} ", "error_ok")
                self._insert(self.txt_predict, "(Good)\n", "error_ok")
            else:
                self._insert(self.txt_predict, f"{error:.1%} ", "error_bad")
                self._insert(self.txt_predict, "(Poor)\n", "error_bad")

        # Right panel
        self.txt_model.delete("1.0", "end")

        self._insert(self.txt_model, "MODEL PERFORMANCE\n", "header")
        self._insert(self.txt_model, "Random Forest Algorithm\n", "muted")
        self._insert(self.txt_model, "=" * 44 + "\n\n", "divider")

        self._insert(self.txt_model, "REGRESSION MODEL\n", "section")
        self._insert(self.txt_model, "-" * 44 + "\n", "divider")
        self._insert(self.txt_model, "Predicts exact win percentage.\n\n", "muted")

        self._insert(self.txt_model, "Mean Absolute Error (MAE):\n", "neutral")
        self._insert(self.txt_model, f"  {mae:.4f}", "positive" if mae < 0.10 else "negative")
        self._insert(self.txt_model, f"  (off by ~{mae:.1%} on average)\n\n", "muted")

        self._insert(self.txt_model, "R-Squared (R²):\n", "neutral")
        self._insert(self.txt_model, f"  {r2:.4f}", "positive" if r2 > 0.3 else "negative")
        self._insert(self.txt_model, f"  (explains {r2:.0%} of variance)\n", "muted")

        self._insert(self.txt_model, "\n", "neutral")

        self._insert(self.txt_model, "CLASSIFICATION MODEL\n", "section")
        self._insert(self.txt_model, "-" * 44 + "\n", "divider")
        self._insert(self.txt_model, "Predicts playoff chances.\n\n", "muted")

        self._insert(self.txt_model, "Accuracy (ACC):\n", "neutral")
        self._insert(self.txt_model, f"  {acc:.1%}", "positive" if acc > 0.55 else "negative")
        self._insert(self.txt_model, f"  (correct {acc:.0%} of the time)\n\n", "muted")

        self._insert(self.txt_model, "Area Under Curve (AUC):\n", "neutral")
        self._insert(self.txt_model, f"  {auc:.4f}", "positive" if auc > 0.6 else "negative")
        self._insert(self.txt_model, f"  (0.5=random, 1.0=perfect)\n", "muted")

        self._insert(self.txt_model, "\n", "neutral")

        self._insert(self.txt_model, "PREDICTION CONFIDENCE\n", "section")
        self._insert(self.txt_model, "-" * 44 + "\n", "divider")
        if prob_top is not None:
            conf_pct = max(prob_top, 1 - prob_top)
            self._insert(self.txt_model, f"Playoff Probability: ", "neutral")
            self._insert(self.txt_model, f"{prob_top:.1%}\n", "neutral")
            self._insert(self.txt_model, f"Model Confidence: ", "neutral")
            conf_tag = "positive" if conf_pct > 0.7 else "neutral" if conf_pct > 0.55 else "negative"
            self._insert(self.txt_model, f"{conf_pct:.0%}\n", conf_tag)

        self._insert(self.txt_model, "\n", "neutral")
        self._insert(self.txt_model, "=" * 44 + "\n", "divider")
        self._insert(self.txt_model, "Trained on first 20 games of each season.\n", "muted")
        self._insert(self.txt_model, "Features: FG, rebounds,\n", "muted")
        self._insert(self.txt_model, "assists, turnovers, and more.\n", "muted")