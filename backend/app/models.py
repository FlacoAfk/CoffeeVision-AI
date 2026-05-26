"""SQLAlchemy models for the CoffeeVision AI backend."""

from datetime import datetime, timezone

from app.extensions import db


class Prediction(db.Model):
    """Stores a single model prediction result.

    Records the prediction made by either the leaf or grain ensemble,
    including the predicted class, confidence score, model type, and
    the original filename for traceability.
    """

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_type = db.Column(
        db.String(10), nullable=False, default="leaf"
    )  # 'leaf' or 'grain'
    top3_json = db.Column(db.Text, nullable=True)  # JSON-encoded top-3 list
    individual_json = db.Column(
        db.Text, nullable=True
    )  # JSON-encoded individual model scores
    processing_time_ms = db.Column(db.Float, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialize the prediction record to a dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "model_type": self.model_type,
            "top_3": self.top3_json,
            "individual": self.individual_json,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<Prediction {self.id}: {self.predicted_class} "
            f"({self.confidence:.3f}) [{self.model_type}]>"
        )