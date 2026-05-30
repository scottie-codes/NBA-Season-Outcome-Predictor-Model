"""
router.py
Handles navigation between screens in the dashboard.
"""

import tkinter as tk

from src.ui.screens.home import HomeScreen
from src.ui.screens.predict import PredictScreen
from src.ui.screens.analysis import AnalysisScreen


class AppRouter:
    """Simple screen router for the dashboard."""

    def __init__(self, root):
        self.root = root
        self.routes = {
            "home": HomeScreen,
            "predict": PredictScreen,
            "analysis": AnalysisScreen,
        }
        self.current_screen = None

    def go(self, screen_name):
        """Navigate to a different screen."""
        if self.current_screen is not None:
            self.current_screen.destroy()

        screen_class = self.routes[screen_name]
        self.current_screen = screen_class(self.root, self)
        self.current_screen.pack(fill="both", expand=True)