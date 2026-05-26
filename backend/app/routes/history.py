"""Blueprint for prediction history endpoint.

GET /api/history
    Returns a paginated list of past predictions, optionally filtered by
    model_type (leaf|grain).

Query parameters:
    page       (int, default=1)     Page number
    per_page   (int, default=20)    Items per page (max 100)
    model_type (str, optional)      Filter: 'leaf' or 'grain'

Response schema:
    {
        "predictions": [ ... ],
        "total": int,
        "page": int,
        "per_page": int,
        "pages": int
    }
"""

from flask import Blueprint, jsonify, request

from app.extensions import db
from app.models import Prediction
from app.schemas import predictions_schema

history_bp = Blueprint("history", __name__, url_prefix="/api")


@history_bp.route("/history", methods=["GET"])
def get_history():
    """Return paginated prediction history with optional model_type filter."""
    # Parse query parameters
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
    except (ValueError, TypeError):
        page = 1
        per_page = 20

    # Clamp per_page to a reasonable range
    per_page = max(1, min(per_page, 100))
    page = max(1, page)

    model_type = request.args.get("model_type", None)

    # Build base query
    query = Prediction.query.order_by(Prediction.created_at.desc())

    if model_type is not None:
        model_type = model_type.strip().lower()
        if model_type in ("leaf", "grain"):
            query = query.filter(Prediction.model_type == model_type)
        else:
            return jsonify({
                "error": f"Invalid model_type '{model_type}'. Use 'leaf' or 'grain'."
            }), 400

    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    predictions = pagination.items
    total = pagination.total
    pages = pagination.pages

    # Serialize
    result = predictions_schema.dump(predictions)

    return jsonify({
        "predictions": result,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }), 200