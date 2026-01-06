import sys
import os

# Add both project root and src directory to Python path
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)

# Add project root to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Add src directory to path
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from gui.main_interface import main

if __name__ == '__main__':
    main()