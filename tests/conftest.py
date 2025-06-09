import os
import sys

# Add the project root directory (parent of 'tests' directory) to sys.path
# This allows pytest to find and import modules from the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
