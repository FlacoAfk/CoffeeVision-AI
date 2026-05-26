"""
Flask application factory for CoffeeVision AI.

Creates and configures the Flask app instance, initializes extensions,
creates database tables, and registers all route blueprints.
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow logs before any TF import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask
from flask_cors import CORS


def create_app(config_object: str = None) -> Flask:
    """Create and configure the CoffeeVision AI Flask application.

    Args:
        config_object: Python dotted path to a config class.
                       Defaults to 'app.config.Config'.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Load configuration
    if config_object is None:
        # Config lives at backend/config.py (sibling to app/ directory)
        # The backend/ directory must be on sys.path for this to work
        from config import Config
    else:
        # Import the provided config path
        mod_path, cls_name = config_object.rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[cls_name])
        Config = getattr(mod, cls_name)

    app.config.from_object(Config)
    app.config.from_pyfile(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instance", "config.py"),
        silent=True,
    )

    # Enable CORS for all routes (React frontend on different origin)
    CORS(app)

    # Initialize extensions
    from app.extensions import db, ma, mongo

    db.init_app(app)
    ma.init_app(app)
    # MongoDB is optional — skip if no MONGO_URI or connection fails
    mongo_uri = app.config.get("MONGO_URI")
    if mongo_uri:
        try:
            mongo.init_app(app, uri=mongo_uri)
        except Exception:
            print("[CoffeeVision AI] MongoDB not available — running without it")

    # Initialize DualModelLoader — real or mock depending on MOCK_MODELS / TF availability
    mock_mode = os.environ.get("MOCK_MODELS", "0") == "1"

    if not mock_mode:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            mock_mode = True
            print("[CoffeeVision AI] TensorFlow not found — auto-enabling mock mode")

    if mock_mode:
        from app.mock_model_loader import MockDualModelLoader
        dual_loader = MockDualModelLoader(config=Config)
        print("[CoffeeVision AI] MOCK_MODELS=1 — using mock predictions (no TensorFlow)")
    else:
        from app.model_loader import DualModelLoader
        dual_loader = DualModelLoader(config=Config)

    app.config["dual_loader"] = dual_loader

    # Create database tables
    with app.app_context():
        from app.models import Prediction  # noqa: F401 — ensure models are imported

        db.create_all()

    # Register blueprints
    from app.routes.predict_leaf import predict_leaf_bp
    from app.routes.predict_grain import predict_grain_bp
    from app.routes.history import history_bp
    from app.routes.iot import iot_bp
    from app.routes.health import health_bp
    from app.routes.models_config import models_bp

    app.register_blueprint(predict_leaf_bp)
    app.register_blueprint(predict_grain_bp)
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(iot_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")

    # Log registered routes for debugging
    if app.debug:
        print(f"\nRegistered routes:")
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
            if methods:
                print(f"  {methods:8s} {rule.rule}")

    print(f"[CoffeeVision AI] Application created successfully")
    return app