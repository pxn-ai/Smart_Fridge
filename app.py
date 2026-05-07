#!/usr/bin/env python3
"""
Top-level launcher for SmartFridge.
This loads the existing Flask app defined in Database/app.py
and runs it so the project can be started from the repository root.
"""
import sys
import os
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure camera_project and Database are importable the same way app.py expects
sys.path.insert(0, str(PROJECT_ROOT / "camera_project"))
sys.path.insert(0, str(PROJECT_ROOT / "camera_project" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Database"))

DB_APP_PATH = PROJECT_ROOT / "Database" / "app.py"

def load_db_app(path: Path):
    spec = importlib.util.spec_from_file_location("smartf.db_app", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    module = load_db_app(DB_APP_PATH)
    # initialize DB if available
    if hasattr(module, 'init_db'):
        try:
            module.init_db()
        except Exception:
            pass

    app = getattr(module, 'app', None)
    if app is None:
        print("Failed to load Flask app from Database/app.py")
        sys.exit(1)

    # Use host 0.0.0.0 so it's accessible on the network
    app.run(host='0.0.0.0', port=5000, debug=False)
