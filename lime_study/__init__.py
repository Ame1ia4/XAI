"""Controlled LIME cost/quality study on Covertype.

Importing the package pins numerical libraries to one thread before numpy loads,
so every entry point (`python -m lime_study....`) gets the same threading.
"""

from lime_study.threads import set_single_thread_env

set_single_thread_env()
