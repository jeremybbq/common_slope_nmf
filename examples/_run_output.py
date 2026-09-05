"""Shared timestamped output-directory handling for executable examples."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_output_dir(
    output_root: Path, *, now: datetime | None = None
) -> Path:
    """Create and return ``output_root / date-time`` for one example run.

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
    output_dir = Path(output_root) / timestamp.strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir
