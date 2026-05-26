"""
DualModelLoader — Singleton that manages leaf and grain prediction models.

Supports two model generations:

  V3 (multi-class, default):
    1. Ensemble mode: loads a single {dataset}_ensemble.keras file
    2. Individual mode (fallback): loads 3 individual .keras models and
       performs weighted soft-voting in Python.
    Leaf: 14 classes, Grain: 3 classes.

  V4 (binary):
    Toggled via Config.V4_MODE = True.
    Uses V4 ensemble paths (leaf_ensemble.keras / grain_ensemble.keras)
    and 2-class name mappings.
    Leaf: ["Sanas", "Roya"], Grain: ["Sano", "Danado"].
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


class DualModelLoader:
    """Singleton managing two prediction pipelines (leaf + grain) with
    lazy loading, ensemble-or-individual auto-detection, and idle eviction."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return
        self._initialized = True

        from config import Config

        self.config = config or Config

        # Model storage — keyed by dataset ("leaf" | "grain")
        # Each entry is either:
        #   {"mode": "ensemble", "model": keras.Model}
        #   {"mode": "individual", "models": {name: keras.Model}, "weights": {name: float}}
        self._loaded = {}  # dataset -> dict

        # Idle tracking
        self._last_used = {}  # dataset -> float (time.time())

        # Class names — V4 binary (roya/broca only)
        self._leaf_class_names = self.config.LEAF_CLASS_NAMES
        self._grain_class_names = self.config.GRAIN_CLASS_NAMES

        # Base directory for resolving relative paths
        self._base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

        print("[DualModelLoader] Initialized. Lazy-load active.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_leaf(self, image: Image.Image) -> dict:
        """Run leaf prediction. Returns dict with predicted_class,
        confidence, top_3, individual."""
        self._last_used["leaf"] = time.time()
        return self._predict_dataset(image, "leaf")

    def predict_grain(self, image: Image.Image) -> dict:
        """Run grain prediction. Returns dict with predicted_class,
        confidence, top_3, individual."""
        self._last_used["grain"] = time.time()
        return self._predict_dataset(image, "grain")

    def predict_single(
        self, image: Image.Image, dataset: str, model_name: str
    ) -> dict:
        """Run a single sub-model prediction for debugging.

        Args:
            image: PIL image to classify.
            dataset: 'leaf' or 'grain'.
            model_name: 'custom_cnn', 'resnet50', or 'efficientnetb0'.
        """
        model = self._load_individual_model(dataset, model_name)
        img_array = self._preprocess_for_model(image, model_name)
        preds = model.predict(img_array, verbose=0)
        class_names = self._get_class_names(dataset)
        return self._format_single(preds[0], class_names, model_name)

    def is_loaded(self, model_type: str) -> bool:
        """Check if models for a given type are loaded in memory."""
        return model_type in self._loaded

    def unload_if_idle(self) -> None:
        """Evict models idle beyond the configured timeout."""
        timeout = self.config.MODEL_IDLE_TIMEOUT
        now = time.time()
        to_evict = [
            ds for ds, t in self._last_used.items()
            if (now - t) > timeout and ds in self._loaded
        ]
        for ds in to_evict:
            print(f"[DualModelLoader] Unloading {ds} models (idle timeout)")
            entry = self._loaded.pop(ds)
            if entry["mode"] == "ensemble":
                del entry["model"]
            else:
                for m in entry["models"].values():
                    del m
            keras.backend.clear_session()

    def reload_weights(self) -> None:
        """Reload model weights and enabled status from model_config.json."""
        import json
        config_path = self._base_dir.parent / "training" / "models" / "model_config.json"
        user_config = {}
        if config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
        
        for dataset in ["leaf", "grain"]:
            if dataset not in self._loaded:
                continue
            entry = self._loaded[dataset]
            if entry["mode"] != "individual":
                continue
            
            domain_config = user_config.get(dataset, {})
            for model_name in list(entry["weights"].keys()):
                mc = domain_config.get(model_name, {})
                if not mc.get("enabled", True):
                    entry["weights"][model_name] = 0.0
                else:
                    w = mc.get("weight", 1.0)
                    entry["weights"][model_name] = max(0.01, float(w))
            
            # Normalize
            total = sum(entry["weights"].values())
            if total > 0:
                for k in entry["weights"]:
                    entry["weights"][k] /= total
            
            print(f"[DualModelLoader] {dataset} weights: {', '.join(f'{k}={v:.3f}' for k, v in entry['weights'].items())}")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _predict_dataset(self, image: Image.Image, dataset: str) -> dict:
        """Run prediction for a dataset, auto-loading models if needed."""
        if dataset not in self._loaded:
            self._load_dataset(dataset)

        entry = self._loaded[dataset]
        class_names = self._get_class_names(dataset)

        if entry["mode"] == "ensemble":
            return self._predict_ensemble(entry["model"], image, class_names)
        else:
            return self._predict_individual(
                entry["models"], entry["weights"], image, class_names
            )

    def _predict_ensemble(
        self, model: keras.Model, image: Image.Image, class_names: list
    ) -> dict:
        """Predict using a single ensemble .keras model."""
        img_array = self._preprocess_ensemble(image)
        preds = model.predict(img_array, verbose=0)
        probs = preds[0]
        return self._format_result(probs, class_names, individual=[])

    def _predict_individual(
        self,
        models: dict,
        weights: dict,
        image: Image.Image,
        class_names: list,
    ) -> dict:
        """Predict with balanced TTA for high accuracy.
        
        3 crops × 2 flips × 3 rotations = 18 preds/model.
        Total: grain=36, leaf=54 predictions. ~5-10 segundos.
        """
        import random
        random.seed(42)
        
        w, h = image.size
        crop_size = min(w, h)
        
        crops = []
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        crops.append(image.crop((left, top, left + crop_size, top + crop_size)))
        if w >= 224 and h >= 224:
            crops.append(image.crop((0, 0, crop_size, crop_size)))
            crops.append(image.crop((w - crop_size, h - crop_size, w, h)))
        
        flips = [None, Image.FLIP_LEFT_RIGHT]
        rotations = [0, 3, -3]
        
        all_probs_accum = []
        individual_results = []
        first_pass = True
        
        for crop_img in crops:
            for flip in flips:
                aug = crop_img.copy()
                if flip is not None:
                    aug = aug.transpose(flip)
                
                for rot in rotations:
                    aug_rot = aug.rotate(rot, expand=False) if rot != 0 else aug
                    
                    all_probs = []
                    for model_name in models:
                        weight = weights.get(model_name, 0)
                        if weight <= 0:
                            continue  # Skip disabled models
                        model = models[model_name]
                        
                        img_array = self._preprocess_for_model(aug_rot, model_name)
                        preds = model.predict(img_array, verbose=0)
                        probs = preds[0]
                        all_probs.append(probs * weight)
                        
                        if first_pass:
                            idx = int(np.argmax(probs))
                            individual_results.append({
                                "model": model_name,
                                "weight": round(weight, 4),
                                "predicted_class": class_names[idx],
                                "confidence": round(float(probs[idx]), 4),
                            })
                    
                    all_probs_accum.append(np.sum(all_probs, axis=0))
                    if first_pass:
                        first_pass = False
        
        combined = np.mean(all_probs_accum, axis=0)
        return self._format_result(combined, class_names, individual=individual_results)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset: str) -> None:
        """Load models for a dataset. Tries ensemble first, falls back
        to individual models with soft-voting."""
        ensemble_path = self._resolve_path(
            self.config.LEAF_MODEL_PATH
            if dataset == "leaf"
            else self.config.GRAIN_MODEL_PATH
        )

        if ensemble_path.exists():
            print(f"[DualModelLoader] Loading {dataset} ensemble from {ensemble_path}")
            model = keras.models.load_model(str(ensemble_path))
            self._loaded[dataset] = {"mode": "ensemble", "model": model}
            print(f"[DualModelLoader] {dataset} ensemble loaded successfully")
        else:
            print(f"[DualModelLoader] No ensemble found for {dataset}, loading individual models")
            models = {}
            model_list = ["custom_cnn", "resnet50", "efficientnetb0"]
            for model_name in model_list:
                model_path = self._get_individual_path(dataset, model_name)
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model not found: {model_path}. "
                        f"Neither ensemble nor individual models available for {dataset}."
                    )
                print(f"[DualModelLoader] Loading {dataset}/{model_name} from {model_path}")
                models[model_name] = keras.models.load_model(str(model_path))

            weights = self._compute_weights(dataset)
            self._loaded[dataset] = {
                "mode": "individual",
                "models": models,
                "weights": weights,
            }
            print(
                f"[DualModelLoader] {dataset} individual models loaded. "
                f"Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}"
            )
            # Apply user config (model_config.json) if it exists
            self.reload_weights()

    def _load_individual_model(self, dataset: str, model_name: str) -> keras.Model:
        """Load a single individual model (for predict_single / debugging)."""
        # If the dataset is already loaded in individual mode, reuse
        if dataset in self._loaded and self._loaded[dataset]["mode"] == "individual":
            return self._loaded[dataset]["models"][model_name]

        path = self._get_individual_path(dataset, model_name)
        if not path.exists():
            raise FileNotFoundError(f"Individual model not found: {path}")
        print(f"[DualModelLoader] Loading individual model: {path}")
        return keras.models.load_model(str(path))

    def _compute_weights(self, dataset: str) -> dict:
        """Compute softmax-normalized weights from val_accuracy metrics."""
        metrics_dir = self._resolve_path(
            self.config.LEAF_MODELS_DIR
            if dataset == "leaf"
            else self.config.GRAIN_MODELS_DIR
        )
        model_keys = ["custom_cnn", "resnet50", "efficientnetb0"]
        accuracies = []

        for model_name in model_keys:
            # Metrics are in: {models_dir}/{dataset}_{model_name}/models/{dataset}_{model_name}_metrics.json
            metric_file = (
                metrics_dir
                / f"{dataset}_{model_name}"
                / "models"
                / f"{dataset}_{model_name}_metrics.json"
            )
            if metric_file.exists():
                with open(str(metric_file)) as f:
                    metrics = json.load(f)
                acc = metrics.get("val_accuracy", 0.5)
                accuracies.append(acc)
                print(f"  {model_name}: val_accuracy = {acc:.4f}")
            else:
                print(f"  Warning: metrics not found for {model_name}, using default 0.5")
                accuracies.append(0.5)

        # Softmax normalization (same formula as build_ensemble.py)
        acc_arr = np.array(accuracies, dtype=np.float32)
        exp_acc = np.exp(acc_arr - np.max(acc_arr))
        weights = exp_acc / np.sum(exp_acc)

        return {k: float(w) for k, w in zip(model_keys, weights)}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_ensemble(image: Image.Image) -> np.ndarray:
        """Preprocess for ensemble model (preprocessing built into graph)."""
        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)

    @staticmethod
    def _preprocess_for_model(image: Image.Image, model_name: str) -> np.ndarray:
        """Preprocess for a specific individual model.

        - custom_cnn: raw pixels [0, 255] (Rescaling layer inside model)
        - resnet50: keras.applications.resnet50.preprocess_input
        - efficientnetb0: keras.applications.efficientnet.preprocess_input
        """
        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        if model_name == "resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
            return preprocess_input(img_array)
        elif model_name == "efficientnetb0":
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input(img_array)

        # custom_cnn: return raw [0, 255]
        return img_array

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(
        probs: np.ndarray, class_names: list, individual: list
    ) -> dict:
        """Format probability vector into API response dict."""
        top_3_indices = np.argsort(probs)[::-1][:3]

        predicted_idx = int(top_3_indices[0])
        predicted_class = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        top_3 = [
            {"class": class_names[int(i)], "confidence": float(probs[int(i)])}
            for i in top_3_indices
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3": top_3,
            "individual": individual,
        }

    @staticmethod
    def _format_single(
        probs: np.ndarray, class_names: list, model_name: str
    ) -> dict:
        """Format a single model prediction."""
        top_3_indices = np.argsort(probs)[::-1][:3]

        predicted_idx = int(top_3_indices[0])
        predicted_class = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        top_3 = [
            {"class": class_names[int(i)], "confidence": float(probs[int(i)])}
            for i in top_3_indices
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3": top_3,
            "individual": [{"model": model_name, "weight": 1.0}],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_class_names(self, dataset: str) -> list:
        if dataset == "leaf":
            return self._leaf_class_names
        return self._grain_class_names

    def _get_individual_path(self, dataset: str, model_name: str) -> Path:
        """Resolve path to an individual model .keras file."""
        models_dir = self._resolve_path(
            self.config.LEAF_MODELS_DIR
            if dataset == "leaf"
            else self.config.GRAIN_MODELS_DIR
        )
        return (
            models_dir
            / f"{dataset}_{model_name}"
            / "models"
            / f"{dataset}_{model_name}.keras"
        )

    def _resolve_path(self, path: str) -> Path:
        """Resolve a potentially relative path against the app base dir."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self._base_dir / p).resolve()

    @property
    def leaf_class_names(self) -> list:
        return self._leaf_class_names

    @leaf_class_names.setter
    def leaf_class_names(self, names: list):
        self._leaf_class_names = names

    @property
    def grain_class_names(self) -> list:
        return self._grain_class_names

    @grain_class_names.setter
    def grain_class_names(self, names: list):
        self._grain_class_names = names
