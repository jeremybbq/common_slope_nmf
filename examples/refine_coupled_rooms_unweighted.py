"""Compatibility shim. Prefer ``experiments.coupled_rooms.refine``."""

from __future__ import annotations

import sys
import warnings

from experiments.coupled_rooms.refine import *  # noqa: F403
from experiments.coupled_rooms.refine import main

_NEW_COMMAND = "python -m experiments.coupled_rooms.refine"
_PLOT_COMMAND = "experiments.coupled_rooms.plot_refine"

def _announce() -> None:
    warnings.warn(
        f"{__name__} is a compatibility shim; run `{_NEW_COMMAND}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


if __name__ == "__main__":
    _announce()
    message = f"redirecting to {_NEW_COMMAND}"
    if _PLOT_COMMAND is not None:
        message += (
            f"\nplot saved NPZ with: python -m {_PLOT_COMMAND} --results PATH"
        )
    print(message, file=sys.stderr)
    main()
