"""Deprecated comparison GUI - use the unified interface in acis.gui.main_interface.

This file remains for backward compatibility. Run the app with:

    python run.py

or import and call:

    from acis.gui.main_interface import main
    main()
"""

def _deprecated():
    raise RuntimeError("Deprecated: please run `python run.py` or use acis.gui.main_interface")