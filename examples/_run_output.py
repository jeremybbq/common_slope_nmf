"""Compatibility shim. Prefer ``experiments._run_output``."""

from experiments._run_output import create_run_output_dir, save_figure

__all__ = ["create_run_output_dir", "save_figure"]
