"""Compatibility shim. Prefer ``experiments.synthetic.demo_synth_init_fit``."""

from __future__ import annotations

import sys
import warnings

from experiments.synthetic.demo_synth_init_fit import *  # noqa: F403
from experiments.synthetic.demo_synth_init_fit import main

_NEW_COMMAND = "python -m experiments.synthetic.demo_synth_init_fit"
_PLOT_COMMAND = "experiments.synthetic.plot_demo_synth_init_fit"

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
