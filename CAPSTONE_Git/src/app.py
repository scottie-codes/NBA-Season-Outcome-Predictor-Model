"""
app.py
Main entry point for the NBA Team Analytics Dashboard GUI.
Application start point (run to open program).
"""

import sys
from pathlib import Path

# Adds project root to path so imports work correctly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from src.ui.router import AppRouter


def main():
    """Starts the Tkinter dashboard app."""
    root = tk.Tk()
    root.title("NBA Team Analytics Dashboard")
    root.geometry("1100x720")

    router = AppRouter(root)
    router.go("home")
    root.mainloop()


if __name__ == "__main__":
    main()