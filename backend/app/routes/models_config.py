"""Model configuration endpoint — adjust weights and enable/disable models."""
import json, os
from pathlib import Path
from flask import Blueprint, jsonify, request

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "training" / "models" / "model_config.json"

DEFAULT_CONFIG = {
    "leaf": {
        "custom_cnn": {"weight": 1.0, "enabled": True},
        "resnet50": {"weight": 1.0, "enabled": True},
        "efficientnetb0": {"weight": 1.0, "enabled": True},
    },
    "grain": {
        "custom_cnn": {"weight": 1.0, "enabled": True},
        "resnet50": {"weight": 1.0, "enabled": True},
        "efficientnetb0": {"weight": 1.0, "enabled": True},
    }
}

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

models_bp = Blueprint("models_config", __name__)

@models_bp.route("/models/config", methods=["GET"])
def get_config():
    return jsonify(load_config())

@models_bp.route("/models/config", methods=["POST"])
def update_config():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    config = load_config()
    
    for domain in ["leaf", "grain"]:
        if domain in data:
            for model_name in ["custom_cnn", "resnet50", "efficientnetb0"]:
                if model_name in data[domain]:
                    model_data = data[domain][model_name]
                    if "weight" in model_data:
                        w = float(model_data["weight"])
                        config[domain][model_name]["weight"] = max(0.01, min(10.0, w))
                    if "enabled" in model_data:
                        config[domain][model_name]["enabled"] = bool(model_data["enabled"])
    
    save_config(config)
    
    # Reload model weights in the running loader
    from app.model_loader import DualModelLoader
    loader = DualModelLoader()
    loader.reload_weights()
    
    return jsonify({"status": "ok", "config": config})
