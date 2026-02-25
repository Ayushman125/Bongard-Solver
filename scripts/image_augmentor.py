import sys
import os

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.bongard_augmentor.cli import main as cli_main

if __name__ == "__main__":
    cli_main()