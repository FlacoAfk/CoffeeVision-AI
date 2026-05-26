"""Marshmallow schemas for serialization and deserialization."""

from marshmallow import fields

from app.extensions import ma
from app.models import Prediction


class PredictionSchema(ma.SQLAlchemyAutoSchema):
    """Schema for the Prediction model with all fields."""

    class Meta:
        model = Prediction
        load_instance = True
        include_fk = True
        fields = (
            "id",
            "filename",
            "predicted_class",
            "confidence",
            "model_type",
            "top_3",
            "individual",
            "processing_time_ms",
            "created_at",
        )

    id = fields.Integer(dump_only=True)
    filename = fields.String(required=True)
    predicted_class = fields.String(required=True)
    confidence = fields.Float(required=True)
    model_type = fields.String(required=True)
    top_3 = fields.Raw(allow_none=True)
    individual = fields.Raw(allow_none=True)
    processing_time_ms = fields.Float(allow_none=True)
    created_at = fields.DateTime(dump_only=True)


prediction_schema = PredictionSchema()
predictions_schema = PredictionSchema(many=True)