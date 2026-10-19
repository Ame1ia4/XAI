"""Force single-threaded numerical libraries so runtimes are comparable across models.

This module must not import numpy (or anything that does): the variables only
take effect if they are set before those libraries load.
"""

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def set_single_thread_env() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def thread_env_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}
