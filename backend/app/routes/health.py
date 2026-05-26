"""Blueprint for health check endpoint.

GET /api/health
    Returns the current service status including model loading state.

Response schema:
    {
        "status": "ok" | "degraded",
        "leaf_loaded": bool,
        "grain_loaded": bool,
        "db_connected": bool,
        "version": "1.0.0",
        "timestamp": "2025-01-01T00:00:00+00:00"
    }
"""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify
from sqlalchemy import text

from app.extensions import db

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Return service health status with model loading and DB connectivity."""
    status = "ok"
    issues = []

    # Check model loader
    dual_loader = current_app.config.get("dual_loader")
    leaf_loaded = dual_loader.is_loaded("leaf") if dual_loader else False
    grain_loaded = dual_loader.is_loaded("grain") if dual_loader else False

    # Check database connectivity
    db_connected = False
    try:
        db.session.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        current_app.logger.warning(f"Health check — DB connection failed: {e}")
        issues.append("database_unreachable")

    # Degrade status if critical components are down
    if not db_connected:
        status = "degraded"
    if not leaf_loaded and not grain_loaded:
        # Not degraded — models lazy-load on first request
        pass

    return jsonify({
        "status": status,
        "leaf_loaded": leaf_loaded,
        "grain_loaded": grain_loaded,
        "db_connected": db_connected,
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issues": issues if issues else None,
    }), 200