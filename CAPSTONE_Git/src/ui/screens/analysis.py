import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def load_data():
    """Load team stats from processed or raw data files."""
    if (PROC / "master_features.csv").exists():
        return pd.read_csv(PROC / "master_features.csv")
    if (RAW / "nba_team_data_regular.csv").exists():
        return pd.read_csv(RAW / "nba_team_data_regular.csv")
    raise FileNotFoundError("No data files found.")


def find_col(df, names):
    """Find column by checking possible name variants."""
    cols = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def season_avg(df, col):
    """Get league average by season for a stat."""
    if col not in df.columns:
        return pd.DataFrame()
    result = df.groupby("Season")[col].mean().reset_index()
    return result.sort_values("Season")


def correlation(df, col1, col2):
    """Calculate correlation between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return None
    clean = df[[col1, col2]].dropna()
    if len(clean) < 3:
        return None
    return clean[col1].corr(clean[col2])


class AnalysisScreen(tk.Frame):
    """Screen showing NBA trends and what stats correlate with winning."""

    def __init__(self, master, router):
        super().__init__(master, bg="white")
        self.router = router
        self.df = None
        self.build_layout()
        self.load_and_display()

    def build_layout(self):
        header = tk.Frame(self, bg="#1a1a2e", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="What Drives Winning in the Modern NBA?",
            font=("Segoe UI", 16, "bold"),
            bg="#1a1a2e",
            fg="white"
        ).pack(side="left", padx=20, pady=15)

        ttk.Button(
            header,
            text="← Back",
            command=lambda: self.router.go("home")
        ).pack(side="right", padx=20, pady=15)

        self.chart_frame = tk.Frame(self, bg="white")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=(5, 0))

        self.insights_frame = tk.Frame(self, bg="#f0f4f8", height=120)
        self.insights_frame.pack(fill="x", padx=10, pady=(5, 10))
        self.insights_frame.pack_propagate(False)

    def load_and_display(self):
        try:
            self.df = load_data()
            self.create_charts()
            self.create_insights()
        except FileNotFoundError as e:
            messagebox.showerror("Data Not Found", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def create_charts(self):
        if self.df is None:
            return

        fig = Figure(figsize=(13, 5), dpi=100, facecolor="white")

        fg3a = find_col(self.df, ["FG3A", "FG3A_base"])
        pts = find_col(self.df, ["PTS", "PTS_base"])
        fg_pct = find_col(self.df, ["FG_PCT", "FG_PCT_base"])
        ast = find_col(self.df, ["AST", "AST_base"])
        tov = find_col(self.df, ["TOV", "TOV_base"])
        win_pct = find_col(self.df, ["TARGET_WIN_PCT", "W_PCT"])

        red = "#e63946"
        teal = "#2a9d8f"
        gold = "#e9c46a"
        blue = "#5dade2"
        orange = "#f5b041"

        # Row 1: League Evolution
        ax1 = fig.add_subplot(231)
        ax1.set_title("3-Point Revolution", fontsize=10, fontweight="bold")
        if fg3a:
            trend = season_avg(self.df, fg3a)
            if not trend.empty:
                x = range(len(trend))
                y = trend[fg3a]
                ax1.plot(x, y, marker="o", linewidth=2, color=red, markersize=4)
                ax1.fill_between(x, y, alpha=0.15, color=red)
                ax1.set_xticks(x[::2])
                ax1.set_xticklabels([s[-5:] for s in trend["Season"].astype(str)][::2], rotation=45, ha="right", fontsize=7)
                ax1.set_ylabel("3PT Attempts", fontsize=8)
                ax1.set_ylim(20, 40)
                change = ((y.iloc[-1] - y.iloc[0]) / y.iloc[0]) * 100
                ax1.text(0.05, 0.92, f"+{change:.0f}%", transform=ax1.transAxes, fontsize=9, fontweight="bold", color=red, va="top")
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("#fafafa")
        ax1.tick_params(labelsize=7)

        ax2 = fig.add_subplot(232)
        ax2.set_title("Scoring Trends", fontsize=10, fontweight="bold")
        if pts:
            trend = season_avg(self.df, pts)
            if not trend.empty:
                x = range(len(trend))
                y = trend[pts]
                ax2.plot(x, y, marker="s", linewidth=2, color=teal, markersize=4)
                ax2.fill_between(x, y, alpha=0.15, color=teal)
                ax2.set_xticks(x[::2])
                ax2.set_xticklabels([s[-5:] for s in trend["Season"].astype(str)][::2], rotation=45, ha="right", fontsize=7)
                ax2.set_ylabel("Points / Game", fontsize=8)
                ax2.set_ylim(95, 120)
                change = ((y.iloc[-1] - y.iloc[0]) / y.iloc[0]) * 100
                ax2.text(0.05, 0.92, f"+{change:.0f}%", transform=ax2.transAxes, fontsize=9, fontweight="bold", color=teal, va="top")
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#fafafa")
        ax2.tick_params(labelsize=7)

        ax3 = fig.add_subplot(233)
        ax3.set_title("Shooting Efficiency", fontsize=10, fontweight="bold")
        if fg_pct:
            trend = season_avg(self.df, fg_pct)
            if not trend.empty:
                x = range(len(trend))
                y = trend[fg_pct]
                ax3.plot(x, y, marker="^", linewidth=2, color=gold, markersize=4)
                ax3.fill_between(x, y, alpha=0.15, color=gold)
                ax3.set_xticks(x[::2])
                ax3.set_xticklabels([s[-5:] for s in trend["Season"].astype(str)][::2], rotation=45, ha="right", fontsize=7)
                ax3.set_ylabel("FG %", fontsize=8)
                ax3.set_ylim(0.44, 0.48)
                ax3.text(0.05, 0.92, f"{y.iloc[-1]:.1%}", transform=ax3.transAxes, fontsize=9, fontweight="bold", color=gold, va="top")
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor("#fafafa")
        ax3.tick_params(labelsize=7)

        # Row 2: What Predicts Winning
        ax4 = fig.add_subplot(234)
        ax4.set_title("Efficiency → Wins", fontsize=10, fontweight="bold")
        if fg_pct and win_pct:
            data = self.df[[fg_pct, win_pct]].dropna()
            if not data.empty:
                x = data[fg_pct]
                y = data[win_pct]
                ax4.scatter(x, y, alpha=0.5, c=blue, edgecolors="#333", linewidths=0.2, s=25)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax4.plot(line_x, p(line_x), "--", color=red, linewidth=1.5)
                r = correlation(self.df, fg_pct, win_pct)
                if r:
                    ax4.text(0.05, 0.92, f"r = {r:.2f}", transform=ax4.transAxes, fontsize=9, fontweight="bold", color=red, va="top")
                ax4.set_xlabel("FG %", fontsize=8)
                ax4.set_ylabel("Win %", fontsize=8)
                ax4.set_xlim(0.40, 0.52)
                ax4.set_ylim(0.15, 0.85)
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor("#fafafa")
        ax4.tick_params(labelsize=7)

        ax5 = fig.add_subplot(235)
        ax5.set_title("Ball Movement → Wins", fontsize=10, fontweight="bold")
        if ast and win_pct:
            data = self.df[[ast, win_pct]].dropna()
            if not data.empty:
                x = data[ast]
                y = data[win_pct]
                ax5.scatter(x, y, alpha=0.5, c=blue, edgecolors="#333", linewidths=0.2, s=25)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax5.plot(line_x, p(line_x), "--", color=red, linewidth=1.5)
                r = correlation(self.df, ast, win_pct)
                if r:
                    ax5.text(0.05, 0.92, f"r = {r:.2f}", transform=ax5.transAxes, fontsize=9, fontweight="bold", color=red, va="top")
                ax5.set_xlabel("Assists / Game", fontsize=8)
                ax5.set_ylabel("Win %", fontsize=8)
                ax5.set_xlim(18, 30)
                ax5.set_ylim(0.15, 0.85)
        ax5.grid(True, alpha=0.3)
        ax5.set_facecolor("#fafafa")
        ax5.tick_params(labelsize=7)

        ax6 = fig.add_subplot(236)
        ax6.set_title("Turnovers → Losses", fontsize=10, fontweight="bold")
        if tov and win_pct:
            data = self.df[[tov, win_pct]].dropna()
            if not data.empty:
                x = data[tov]
                y = data[win_pct]
                ax6.scatter(x, y, alpha=0.5, c=orange, edgecolors="#333", linewidths=0.2, s=25)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax6.plot(line_x, p(line_x), "--", color=red, linewidth=1.5)
                r = correlation(self.df, tov, win_pct)
                if r:
                    ax6.text(0.05, 0.92, f"r = {r:.2f}", transform=ax6.transAxes, fontsize=9, fontweight="bold", color=red, va="top")
                ax6.set_xlabel("Turnovers / Game", fontsize=8)
                ax6.set_ylabel("Win %", fontsize=8)
                ax6.set_xlim(10, 18)
                ax6.set_ylim(0.15, 0.85)
        ax6.grid(True, alpha=0.3)
        ax6.set_facecolor("#fafafa")
        ax6.tick_params(labelsize=7)

        fig.text(0.5, 0.98, "LEAGUE EVOLUTION (2015-2024)", ha="center", fontsize=10, fontweight="bold", color="#1a1a2e")
        fig.text(0.5, 0.49, "WHAT PREDICTS WINNING?", ha="center", fontsize=10, fontweight="bold", color="#1a1a2e")
        fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.55, wspace=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_insights(self):
        if self.df is None:
            return

        fg3a = find_col(self.df, ["FG3A", "FG3A_base"])
        fg_pct = find_col(self.df, ["FG_PCT", "FG_PCT_base"])
        ast = find_col(self.df, ["AST", "AST_base"])
        tov = find_col(self.df, ["TOV", "TOV_base"])
        win_pct = find_col(self.df, ["TARGET_WIN_PCT", "W_PCT"])

        fg3_change = None
        if fg3a:
            trend = season_avg(self.df, fg3a)
            if len(trend) >= 2:
                fg3_change = ((trend[fg3a].iloc[-1] - trend[fg3a].iloc[0]) / trend[fg3a].iloc[0]) * 100

        r_fg = correlation(self.df, fg_pct, win_pct) if fg_pct and win_pct else None
        r_ast = correlation(self.df, ast, win_pct) if ast and win_pct else None
        r_tov = correlation(self.df, tov, win_pct) if tov and win_pct else None

        seasons = self.df["Season"].nunique() if "Season" in self.df.columns else 0
        records = len(self.df)

        left_frame = tk.Frame(self.insights_frame, bg="#f0f4f8")
        left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)

        right_frame = tk.Frame(self.insights_frame, bg="#f0f4f8")
        right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=10)

        # Left side, Key Findings
        tk.Label(
            left_frame,
            text="KEY FINDINGS",
            font=("Segoe UI", 10, "bold"),
            bg="#f0f4f8",
            fg="#1a1a2e"
        ).pack(anchor="w")

        findings = []
        if fg3_change:
            findings.append(f"• 3 PT REVOLUTION: 3PT attempts up {fg3_change:.0f}% since 2015.")
        if r_fg:
            findings.append(f"• EFFICIENCY (GOOD SHOTS) WIN: FG% is the strongest predictor of winning (r = {r_fg:.2f})")
        if r_ast:
            findings.append(f"• PASSING = BETTER SHOTS = MORE WINS: Ball movement leads to winning. (r = {r_ast:.2f})")
        if r_tov:
            findings.append(f"• PROTECT THE BALL: Turnovers hurt winning. (r = {r_tov:.2f})")

        tk.Label(
            left_frame,
            text="\n".join(findings),
            font=("Segoe UI", 9),
            bg="#f0f4f8",
            fg="#333",
            justify="left"
        ).pack(anchor="w", pady=(5, 0))

        # Right side, connect to my model
        tk.Label(
            right_frame,
            text="HOW THIS CONNECTS TO PREDICTIONS",
            font=("Segoe UI", 10, "bold"),
            bg="#f0f4f8",
            fg="#1a1a2e"
        ).pack(anchor="w")

        connection_text = (
            f"These and other correlations from key stats inform our NBA Season Outcome Predictor model. \n"
             f"The model uses the first 20 games from each season to predict final win %.\n"
            f"Data: {seasons} seasons, {records} team records (2015-2024)"
        )

        tk.Label(
            right_frame,
            text=connection_text,
            font=("Segoe UI", 9),
            bg="#f0f4f8",
            fg="#333",
            justify="left"
        ).pack(anchor="w", pady=(5, 0))