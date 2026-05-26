"""V4 binary kernels: push 6 notebooks (3 leaf + 3 grain) across 3 accounts.

Binary classes:
  - leaf: Sanas (healthy) vs Enfermedad-Roya (rust disease)
  - grain: Sano (healthy) vs Danado (damaged)

Training strategy:
  - Heavy augmentation + train_ds.repeat(10) for 10x effective size
  - 2-phase: 20 epochs frozen → 30 epochs fine-tune top 30 layers
  - Class weights computed from training distribution
  - Output: Dense(2, softmax)
"""
import os, json, tempfile, shutil

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_5e205dcd6f68c3b4ff4c4c121a149bf8",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
    "juanveru": "KGAT_fcb1ba0670cfb603b1a1ab6053eb7961",
}

MOUNT_PATHS = {
    "leaf": {
        "julianmedinamonje45": "/kaggle/input/datasets/julianmedinamonje45/coffevision-leaf-cls",
        "juanveru": "/kaggle/input/datasets/juanveru/coffevision-leaf-cls",
    },
    "grain": {
        "vann234": "/kaggle/input/datasets/vann234/coffevision-grain-cls",
        "juanveru": "/kaggle/input/datasets/juanveru/coffevision-grain-cls",
    },
}

LEAF_CLASS_NAMES = ["Sanas", "Enfermedad-Roya"]
GRAIN_CLASS_NAMES = ["Sano", "Danado"]

KERNELS = [
    # julianmedinamonje45 — leaf (ResNet50 + Custom CNN)
    {"dataset": "leaf", "model": "custom_cnn", "username": "julianmedinamonje45",
     "slug": "coffeevision-leaf-customcnn-v4", "title": "CoffeeVision Leaf Custom CNN V4",
     "num_classes": 2, "class_names": LEAF_CLASS_NAMES,
     "dataset_slug": "julianmedinamonje45/coffevision-leaf-cls"},
    {"dataset": "leaf", "model": "resnet50", "username": "julianmedinamonje45",
     "slug": "coffeevision-leaf-resnet50-v4", "title": "CoffeeVision Leaf ResNet50 V4",
     "num_classes": 2, "class_names": LEAF_CLASS_NAMES,
     "dataset_slug": "julianmedinamonje45/coffevision-leaf-cls"},
    # vann234 — grain (ResNet50 + Custom CNN)
    {"dataset": "grain", "model": "custom_cnn", "username": "vann234",
     "slug": "coffeevision-grain-customcnn-v4", "title": "CoffeeVision Grain Custom CNN V4",
     "num_classes": 2, "class_names": GRAIN_CLASS_NAMES,
     "dataset_slug": "vann234/coffevision-grain-cls"},
    {"dataset": "grain", "model": "resnet50", "username": "vann234",
     "slug": "coffeevision-grain-resnet50-v4", "title": "CoffeeVision Grain ResNet50 V4",
     "num_classes": 2, "class_names": GRAIN_CLASS_NAMES,
     "dataset_slug": "vann234/coffevision-grain-cls"},
    # juanveru — leaf EfficientNet + grain EfficientNet
    {"dataset": "leaf", "model": "efficientnetb0", "username": "juanveru",
     "slug": "coffeevision-leaf-effnet-v4", "title": "CoffeeVision Leaf EfficientNet V4",
     "num_classes": 2, "class_names": LEAF_CLASS_NAMES,
     "dataset_slug": "juanveru/coffevision-leaf-cls"},
    {"dataset": "grain", "model": "efficientnetb0", "username": "juanveru",
     "slug": "coffeevision-grain-effnet-v4", "title": "CoffeeVision Grain EfficientNet V4",
     "num_classes": 2, "class_names": GRAIN_CLASS_NAMES,
     "dataset_slug": "juanveru/coffevision-grain-cls"},
]


def generate_notebook(kcfg):
    num_classes = kcfg["num_classes"]
    class_names_str = json.dumps(kcfg["class_names"])
    dataset = kcfg["dataset"]
    model = kcfg["model"]
    username = kcfg["username"]
    data_dir = MOUNT_PATHS[dataset][username]

    cell1 = f"""import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import json

print(f"TF version: {{tf.__version__}}")
print(f"GPU: {{tf.config.list_physical_devices('GPU')}}")

CLASS_NAMES = {class_names_str}
NUM_CLASSES = {num_classes}
IMG_SIZE = 224
BATCH_SIZE = 32
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 30
DATASET = "{dataset}"
MODEL_NAME = "{model}"
DATA_DIR = "{data_dir}"
"""

    cell2 = """# Load images from class folders, filter to binary target classes
from pathlib import Path

data_dir = Path(DATA_DIR)
print(f"Data dir: {data_dir}")
print(f"Exists: {data_dir.exists()}")
if data_dir.exists():
    dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Class dirs found ({len(dirs)}): {dirs}")

records = []
for cls_dir in sorted(data_dir.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    if cls_name not in CLASS_NAMES:
        continue
    for img_path in cls_dir.iterdir():
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            records.append({
                'filepath': str(img_path),
                'label': cls_name,
            })

df = pd.DataFrame(records)
print(f"Total images: {len(df)}")
print(f"Class distribution:\\n{df['label'].value_counts()}")

label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
df['label_idx'] = df['label'].map(label_to_idx)

# Stratified split: 70-15-15
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss1.split(df, df['label_idx']))
df_train = df.iloc[train_idx]
df_temp = df.iloc[temp_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(sss2.split(df_temp, df_temp['label_idx']))
df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
print(f"Train class dist:\\n{df_train['label'].value_counts().sort_index()}")
print(f"Val class dist:\\n{df_val['label'].value_counts().sort_index()}")
print(f"Test class dist:\\n{df_test['label'].value_counts().sort_index()}")

cw = compute_class_weight('balanced', classes=np.array(CLASS_NAMES), y=df_train['label'].values)
class_weight = {i: w for i, w in enumerate(cw)}
print(f"Class weights: {class_weight}")

for name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
    split_df.to_csv(f'/kaggle/working/{DATASET}_{name}.csv', index=False)
"""

    cell3 = """# Build tf.data pipelines with HEAVY augmentation

def parse_image(filepath, label_idx):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label_idx

AUTOTUNE = tf.data.AUTOTUNE

# Heavy augmentation for V4 binary training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

# Train pipeline: parse → cache → augment → repeat(10) → shuffle → batch → prefetch
train_ds = tf.data.Dataset.from_tensor_slices(
    (df_train['filepath'].values, df_train['label_idx'].values)
).map(parse_image, num_parallel_calls=AUTOTUNE).cache()
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.repeat(10)
train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (df_val['filepath'].values, df_val['label_idx'].values)
).map(parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (df_test['filepath'].values, df_test['label_idx'].values)
).map(parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print(f"Train batches per epoch: {len(train_ds)}")
print(f"Val batches: {len(val_ds)}")
print(f"Test batches: {len(test_ds)}")
"""

    if model == "custom_cnn":
        cell4 = f"""# Build Custom CNN (no pretrained base — single phase training)
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense({num_classes}, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name="custom_cnn_v4")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
"""
    elif model == "resnet50":
        cell4 = f"""# Build ResNet50 transfer learning (2-phase training)
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense({num_classes}, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name="resnet50_v4")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
"""
    else:
        cell4 = f"""# Build EfficientNetB0 transfer learning (2-phase training)
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense({num_classes}, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name="efficientnetb0_v4")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
"""

    if model == "custom_cnn":
        cell5 = f"""# Train Custom CNN — single phase (50 epochs)
os.makedirs('/kaggle/working/models', exist_ok=True)
save_path = f'/kaggle/working/models/{{DATASET}}_{{MODEL_NAME}}.keras'

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
)

model.save(save_path)
"""
    else:
        cell5 = f"""# Phase 1: Frozen backbone — 20 epochs
os.makedirs('/kaggle/working/models', exist_ok=True)
save_path = f'/kaggle/working/models/{{DATASET}}_{{MODEL_NAME}}.keras'

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]

print("=" * 60)
print("PHASE 1: Training with frozen backbone")
print("=" * 60)
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=callbacks_phase1,
    class_weight=class_weight,
)

# Phase 2: Unfreeze top 30 layers — fine-tune for 30 epochs
print("=" * 60)
print("PHASE 2: Fine-tuning top 30 layers")
print("=" * 60)

base_model.trainable = True
# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=callbacks_phase2,
    class_weight=class_weight,
)

model.save(save_path)
"""

    cell6 = f"""# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {{test_acc:.4f}}")
print(f"Test loss: {{test_loss:.4f}}")

# Collect all val accuracies from both phases
val_accs = []
if 'val_accuracy' in history.history:
    val_accs = history.history['val_accuracy']
else:
    val_accs = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

metrics = {{
    'val_accuracy': float(max(val_accs)),
    'val_loss': float(min(history.history['val_loss'])) if 'val_loss' in history.history
                 else float(min(history_phase1.history['val_loss'] + history_phase2.history['val_loss'])),
    'best_epoch': int(np.argmax(val_accs)),
    'total_epochs': len(val_accs),
    'final_train_acc': float(history.history['accuracy'][-1]) if 'accuracy' in history.history
                        else float(history_phase2.history['accuracy'][-1]),
    'final_val_acc': float(val_accs[-1]),
}}
with open(f'/kaggle/working/models/{{DATASET}}_{{MODEL_NAME}}_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\\nTraining complete!")
print(f"Best val_accuracy: {{metrics['val_accuracy']:.4f}} at epoch {{metrics['best_epoch']}}")
print(f"Model saved to: {{save_path}}")

test_metrics = {{
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
}}
with open(f'/kaggle/working/models/{{DATASET}}_{{MODEL_NAME}}_test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# Per-class accuracy
from sklearn.metrics import classification_report
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print(f"\\nClassification Report:\\n{{report}}")

with open(f'/kaggle/working/models/{{DATASET}}_{{MODEL_NAME}}_classification_report.txt', 'w') as f:
    f.write(report)

print("Done! All metrics and models saved to /kaggle/working/models/")
"""

    cells = []
    for src, ctype in [(cell1, "code"), (cell2, "code"), (cell3, "code"),
                        (cell4, "code"), (cell5, "code"), (cell6, "code")]:
        cells.append({
            "cell_type": ctype, "metadata": {}, "source": [src + "\n"],
            "outputs": [], "execution_count": None,
            "id": os.urandom(8).hex(),
        })

    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "cells": cells,
    }


def push_kernel(token, username, slug, title, notebook, dataset_slug):
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    tmp = tempfile.mkdtemp(prefix="coffevision_v4_")
    try:
        with open(os.path.join(tmp, "notebook.ipynb"), "w") as f:
            json.dump(notebook, f)
        meta = {
            "id": f"{username}/{slug}", "title": title,
            "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true",
            "enable_gpu": "true", "enable_internet": "true",
            "dataset_sources": [dataset_slug],
        }
        with open(os.path.join(tmp, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Pushing {slug}...", end=" ", flush=True)
        api.kernels_push(tmp)
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 70)
    print("CoffeeVision AI — V4 Binary Kernels")
    print("=" * 70)
    for kcfg in KERNELS:
        print(f"\n  {kcfg['dataset']} / {kcfg['model']} -> {kcfg['username']}")
        print(f"  DATA_DIR = {MOUNT_PATHS[kcfg['dataset']][kcfg['username']]}")
        print(f"  Split = StratifiedShuffleSplit (70-15-15)")
        notebook = generate_notebook(kcfg)
        push_kernel(
            token=KAGGLE_TOKENS[kcfg["username"]],
            username=kcfg["username"],
            slug=kcfg["slug"],
            title=kcfg["title"],
            notebook=notebook,
            dataset_slug=kcfg["dataset_slug"],
        )

    print("\n" + "=" * 70)
    print("All V4 kernels pushed! URLs:")
    for kcfg in KERNELS:
        url = f"https://www.kaggle.com/code/{kcfg['username']}/{kcfg['slug']}"
        print(f"  {kcfg['dataset']:5s}/{kcfg['model']:15s} -> {url}")
    print("=" * 70)
