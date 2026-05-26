"""Blueprint for IoT endpoint (ESP32-CAM integration).

POST /api/iot/data
    Accepts an image payload from ESP32-CAM, routes it through the
    leaf ensemble internally, and returns the prediction result.

The IoT device (ESP32-CAM) posts images captured in the field. Since
the grain analysis is manual upload only (farmers inspect grains at
the mill), all IoT traffic goes to the leaf ensemble.

Response schema:
    {
        "stored": true,
        "sensor_reading_id": int,
        "prediction": {
            "predicted_class": str,
            "confidence": float,
            "top_3": [...],
            "model_type": "leaf"
        }
    }
"""

import io
import json
import time
from datetime import datetime, timezone

from PIL import Image
from flask import Blueprint, current_app, jsonify, request

from app.extensions import db, mongo
from app.models import Prediction

iot_bp = Blueprint("iot", __name__, url_prefix="/api/iot")


@iot_bp.route("/data", methods=["POST"])
def iot_data():
    """Receive image from ESP32-CAM, run leaf prediction, return result."""
    # Accept image from various input formats the ESP32-CAM may use
    image_data = None

    if "image" in request.files:
        image_data = request.files["image"].read()
    elif request.content_type and "image" in request.content_type:
        image_data = request.get_data()
    elif request.is_json:
        body = request.get_json()
        if body and "image" in body:
            import base64

            image_data = base64.b64decode(body["image"])

    if image_data is None:
        return jsonify({"error": "No image data received. Send as multipart/form-data, raw image, or base64 JSON."}), 400

    # Validate and decode image
    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        image = Image.open(io.BytesIO(image_data))
    except Exception:
        return jsonify({"error": "Invalid image data from IoT device."}), 400

    # Run leaf prediction
    try:
        start_time = time.time()

        dual_loader = current_app.config["dual_loader"]
        result = dual_loader.predict_leaf(image)

        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        # Persist to SQLite
        prediction = Prediction(
            filename=f"iot_capture_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg",
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            model_type="leaf",
            top3_json=json.dumps(result["top_3"]),
            individual_json=json.dumps(result.get("individual", [])),
            processing_time_ms=processing_time_ms,
        )
        db.session.add(prediction)
        db.session.commit()

        # Also persist to MongoDB for IoT-specific analytics
        try:
            mongo.db.iot_readings.insert_one({
                "prediction_id": prediction.id,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "model_type": "leaf",
                "top_3": result["top_3"],
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc),
            })
        except Exception as mongo_err:
            current_app.logger.warning(
                f"MongoDB insert failed (non-fatal): {mongo_err}"
            )

        return jsonify({
            "stored": True,
            "sensor_reading_id": prediction.id,
            "prediction": {
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "top_3": result["top_3"],
                "individual": result.get("individual", []),
                "model_type": "leaf",
                "processing_time_ms": processing_time_ms,
            },
        }), 200

    except Exception as e:
        current_app.logger.error(f"IoT prediction error: {e}", exc_info=True)
        return jsonify({"error": f"IoT prediction failed: {str(e)}"}), 500