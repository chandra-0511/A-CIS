"""Deprecated helper module - use the unified application at project root.

This file kept for backward compatibility. Prefer running `python run.py` or
calling `acis.gui.main_interface.main()`.
"""

def start_cache_comparison(*args, **kwargs):
    raise RuntimeError("start_cache_comparison is deprecated; run `python run.py` instead or import acis.gui.main_interface")