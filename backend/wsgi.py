"""
WSGI entry point for CoffeeVision AI.

This is the file that Gunicorn (or any WSGI server) loads to serve
the Flask application in production.

Usage:
    gunicorn wsgi:app
    python wsgi.py            # Development server
"""

import os
import sys

# Ensure the backend directory is on the Python path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    print(f"[CoffeeVision AI] Starting development server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)