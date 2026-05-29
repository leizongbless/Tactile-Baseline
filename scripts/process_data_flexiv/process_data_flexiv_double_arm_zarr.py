"""Deprecated: use process_data_flexiv_zarr.py with --dual_arm.

Kept as a thin compatibility shim so historical shell scripts and notes
continue to function. Forwards all CLI args to process_data_flexiv_zarr.py
with --dual_arm injected and prints a deprecation notice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


def main():
    warnings.warn(
        "process_data_flexiv_double_arm_zarr.py is deprecated; "
        "use process_data_flexiv_zarr.py --dual_arm.",
        DeprecationWarning,
        stacklevel=2,
    )

    script = Path(__file__).with_name("process_data_flexiv_zarr.py")
    new_argv = [sys.executable, str(script), "--dual_arm", *sys.argv[1:]]
    os.execv(new_argv[0], new_argv)


if __name__ == "__main__":
    main()
