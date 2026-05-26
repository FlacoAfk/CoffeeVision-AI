"""
Mock DualModelLoader for local development without TensorFlow.

Returns deterministic fake predictions matching the real CoffeeVision API contract.
Activate by setting environment variable: MOCK_MODELS=1

Usage:
  MOCK_MODELS=1 python wsgi.py
"""

import random
import time

from PIL import Image

LEAF_CLASS_NAMES = [
    "DeficienciaNutricional-Boro",
    "DeficienciaNutricional-Calcio",
    "DeficienciaNutricional-Fosforo",
    "DeficienciaNutricional-Hierro",
    "DeficienciaNutricional-Magnesio",
    "DeficienciaNutricional-Manganeso",
    "DeficienciaNutricional-Nitrogeno",
    "DeficienciaNutricional-Potasio",
    "Enfermedad-Antracnosis",
    "Enfermedad-Mancha-de-hierro",
    "Enfermedad-Roya",
    "Plaga-AranaRoja",
    "Plaga-Minador",
    "Sanas",
]

GRAIN_CLASS_NAMES = ["Danado", "Infectado", "Sano"]


class MockDualModelLoader:
    """Mock dual loader that returns fake predictions without TensorFlow."""

    _instance = None

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return
        self._initialized = True
        self._leaf_class_names = LEAF_CLASS_NAMES
        self._grain_class_names = GRAIN_CLASS_NAMES
        print("[MockDualModelLoader] Initialized (mock mode — no TensorFlow)")

    def predict_leaf(self, image: Image.Image) -> dict:
        return self._predict_dataset(image, "leaf")

    def predict_grain(self, image: Image.Image) -> dict:
        return self._predict_dataset(image, "grain")

    def predict_single(self, image, dataset, model_name):
        result = self._predict_dataset(image, dataset)
        return {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "top_3": result["top_3"],
            "individual": [{"model": model_name, "weight": 1.0}],
        }

    def is_loaded(self, model_type):
        return True

    def unload_if_idle(self):
        pass

    def _predict_dataset(self, image, dataset):
        random.seed(hash(image.size) + hash(dataset))

        if dataset == "leaf":
            classes = self._leaf_class_names
            # ~30% chance of "Sanas" (healthy)
            if random.random() < 0.3:
                top_class = "Sanas"
            else:
                disease = [c for c in classes if c != "Sanas"]
                top_class = random.choice(disease)
        else:
            classes = self._grain_class_names
            top_class = random.choice(classes)

        confidence = round(random.uniform(0.75, 0.97), 4)
        remaining = [c for c in classes if c != top_class]
        top_2 = random.sample(remaining, min(2, len(remaining)))

        top_3 = [{"class": top_class, "confidence": confidence}]
        for c in top_2:
            top_3.append({"class": c, "confidence": round(random.uniform(0.01, 0.12), 4)})

        weights = {"custom_cnn": 0.28, "resnet50": 0.36, "efficientnetb0": 0.36}
        individual = []
        for name, w in weights.items():
            individual.append({
                "model": name,
                "weight": round(w, 4),
                "predicted_class": top_class,
                "confidence": round(confidence * random.uniform(0.9, 1.0), 4),
            })

        time.sleep(0.05)

        return {
            "predicted_class": top_class,
            "confidence": confidence,
            "top_3": top_3,
            "individual": individual,
        }

    @property
    def leaf_class_names(self):
        return self._leaf_class_names

    @leaf_class_names.setter
    def leaf_class_names(self, names):
        self._leaf_class_names = names

    @property
    def grain_class_names(self):
        return self._grain_class_names

    @grain_class_names.setter
    def grain_class_names(self, names):
        self._grain_class_names = names
