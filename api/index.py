import os
import sys

# Add the project root and backend/app to the Python path
# This allows 'from backend.app.main import app' and 'from app... import ...' to work
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "backend"))
sys.path.append(os.path.join(root_dir, "backend", "app"))

from backend.app.main import app