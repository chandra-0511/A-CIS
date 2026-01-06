#!/usr/bin/env python3
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from acis.gui.main_interface import main

if __name__ == '__main__':
    main()