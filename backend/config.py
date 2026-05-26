import os


class Config:
    """Application configuration loaded from environment variables."""

    # Individual model directories (V4 binary models — roya/broca only)
    LEAF_MODELS_DIR = os.environ.get(
        "LEAF_MODELS_DIR",
        r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\models\v4",
    )
    GRAIN_MODELS_DIR = os.environ.get(
        "GRAIN_MODELS_DIR",
        r"C:\Users\elkaw\Desktop\CoffeeVision AI\training\models\v5_all",
    )

    # Ensemble model paths (empty -> use individual soft-voting)
    LEAF_MODEL_PATH = os.environ.get(
        "LEAF_MODEL_PATH",
        r"__no_ensemble__",
    )
    GRAIN_MODEL_PATH = os.environ.get(
        "GRAIN_MODEL_PATH",
        r"__no_ensemble__",
    )

    # Ensemble metadata files (weights, class mappings)
    LEAF_METADATA_PATH = os.environ.get(
        "LEAF_METADATA_PATH",
        "",
    )
    GRAIN_METADATA_PATH = os.environ.get(
        "GRAIN_METADATA_PATH",
        "",
    )

    # Model lifecycle
    MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "600"))

    # Leaf class names (2 classes — V4 binary: roya detection)
    LEAF_CLASS_NAMES = [
        "Sanas",
        "Roya",
    ]

    # Grain class names (2 classes — V4 binary: broca detection)
    GRAIN_CLASS_NAMES = [
        "Sano",
        "Danado",
    ]

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///predictions.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # MongoDB
    MONGO_URI = os.environ.get(
        "MONGO_URI", "mongodb://localhost:27017/coffevision"
    )

    # Security
    SECRET_KEY = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )

    # Image processing
    IMG_SIZE = 224

    # Model names for individual predictions
    MODEL_NAMES = ["custom_cnn", "resnet50", "efficientnetb0"]
