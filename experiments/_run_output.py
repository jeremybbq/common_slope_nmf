"""Timestamped output directories and figure file I/O for experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def create_run_output_dir(
    output_root: Path, *, now: datetime | None = None
) -> Path:
    """Create and return ``output_root / date-time`` for one experiment run.

    Parameters
    ----------
    output_root
        Parent directory beneath which the unique run directory is created.
    now
        Optional timestamp used for deterministic tests. When omitted, use the
        current local timezone and include microseconds to avoid collisions.

    Returns
    -------
    Path
        Created run directory named ``YYYY-MM-DD_HH-MM-SS-ffffff``.
    """

    timestamp = datetime.now().astimezone() if now is None else now
    output_dir = Path(output_root) / timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_figure(
    figure: Figure,
    output_dir: Path,
    filename: str,
    *,
    show: bool = False,
    dpi: int = 180,
) -> Path:
    """Save a figure as PNG (and PDF when the suffix is ``.png``) and return the PNG path."""

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as MatplotlibFigure

    if not isinstance(figure, MatplotlibFigure):
        raise TypeError("figure must be a Matplotlib Figure.")

    path = Path(output_dir) / filename
    figure.savefig(path, dpi=dpi)
    if path.suffix.lower() == ".png":
        figure.savefig(path.with_suffix(".pdf"))
    if not show:
        plt.close(figure)
    return path.resolve()
