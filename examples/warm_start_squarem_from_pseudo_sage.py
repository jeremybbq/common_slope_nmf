"""Compatibility shim. Prefer ``experiments.squarem.warm_start``."""

from __future__ import annotations

import sys
import warnings

from experiments.squarem.warm_start import *  # noqa: F403
from experiments.squarem.warm_start import main

_NEW_COMMAND = "python -m experiments.squarem.warm_start"
_PLOT_COMMAND = "experiments.squarem.plot_warm_start"

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
