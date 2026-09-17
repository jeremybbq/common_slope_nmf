"""Compatibility shim. Prefer ``experiments.room_to_hallway.omni``."""

from __future__ import annotations

import sys
import warnings

from experiments.room_to_hallway.omni import *  # noqa: F403
from experiments.room_to_hallway.omni import main

_NEW_COMMAND = "python -m experiments.room_to_hallway.omni"
_PLOT_COMMAND = "experiments.room_to_hallway.plot_omni"

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
