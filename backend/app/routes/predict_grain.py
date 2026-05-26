"""Blueprint for grain prediction endpoint.

POST /api/predict/grain
    Accepts a multipart image upload, runs grain ensemble inference,
    persists the result, and returns the prediction.

Response schema:
    {
        "id": int,
        "predicted_class": str,
        "confidence": float,
        "top_3": [{"class": str, "confidence": float}, ...],
        "individual": [{"model": str, "weight": float}, ...],
        "model_type": "grain",
        "processing_time_ms": float
    }
"""

import io
import json
import time

from PIL import Image
from flask import Blueprint, current_app, jsonify, request

from app.extensions import db
from app.models import Prediction

predict_grain_bp = Blueprint("predict_grain", __name__, url_prefix="/api/predict")


@predict_grain_bp.route("/grain", methods=["POST"])
def predict_grain():
    """Handle grain quality prediction from an uploaded image."""
    # Validate request has a file
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Validate file is a valid image
    try:
        file.seek(0)
        image = Image.open(io.BytesIO(file.read()))
        image.verify()
        file.seek(0)
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file. Upload a valid JPEG, PNG, or WEBP."}), 400

    # Run prediction
    try:
        start_time = time.time()

        dual_loader = current_app.config["dual_loader"]
        result = dual_loader.predict_grain(image)

        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        # Persist prediction to database
        prediction = Prediction(
            filename=file.filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            model_type="grain",
            top3_json=json.dumps(result["top_3"]),
            individual_json=json.dumps(result.get("individual", [])),
            processing_time_ms=processing_time_ms,
        )
        db.session.add(prediction)
        db.session.commit()

        # Build response
        response = {
            "id": prediction.id,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "top_3": result["top_3"],
            "individual": result.get("individual", []),
            "model_type": "grain",
            "processing_time_ms": processing_time_ms,
        }
        return jsonify(response), 200

    except Exception as e:
        current_app.logger.error(f"Grain prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500