"""Push V5 with kagglehub authentication."""
import os, json, tempfile, shutil

TOKEN = "KGAT_b4a099147090d704a99172922f836189"
os.environ["KAGGLE_API_TOKEN"] = TOKEN
os.environ.pop("KAGGLE_USERNAME", None); os.environ.pop("KAGGLE_KEY", None)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = r'''# %% [markdown]
# # __TITLE__

# %% 
import os, json, subprocess, sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
print("TF:", tf.__version__)

# %% 
# === DOWNLOAD DATASETS (with auth) ===
# Set Kaggle credentials for kagglehub
os.environ["KAGGLE_USERNAME"] = "juanestebanroja"
os.environ["KAGGLE_KEY"] = "b4a099147090d704a99172922f836189"

subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"], check=False)
import kagglehub

broca_path = kagglehub.dataset_download("vann234/coffevision-grain-broca-v5")
sano_path = kagglehub.dataset_download("vann234/coffevision-grain-cls")
print(f"Broca: {broca_path}")
print(f"Sano: {sano_path}")

# %% 
CLASS_NAMES = ["Sano", "Danado"]
NUM_CLASSES = 2; IMG_SIZE = 224; BATCH_SIZE = 32; SEED = 42

def load_files(d, lbl):
    fs, ls = [], []
    if not os.path.isdir(d): return fs, ls
    for r, _, fns in os.walk(d):
        for f in fns:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                fs.append(os.path.join(r, f)); ls.append(lbl)
    return fs, ls

sf, sl = load_files(os.path.join(sano_path, "Sano"), 0)
print(f"Sano: {len(sf)}")

df, dl = [], []
for r, dirs, _ in os.walk(broca_path):
    for d in dirs:
        if d.lower() in ["danado", "broca", "broca de cafe"]:
            f, l = load_files(os.path.join(r, d), 1)
            df.extend(f); dl.extend(l)
            print(f"Danado ({d}): {len(f)}")
if not df:
    for fn in os.listdir(broca_path):
        if fn.lower().endswith(('.jpg','.jpeg','.png')):
            df.append(os.path.join(broca_path, fn)); dl.append(1)
    print(f"Danado (root): {len(df)}")

af, al = sf + df, sl + dl
print(f"Total: {len(af)} images")

# %% 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
ti, tpi = next(sss.split(af, al))
tl2 = [al[i] for i in tpi]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
vri, tri = next(sss2.split(range(len(tpi)), tl2))
vi = [tpi[i] for i in vri]; tei = [tpi[i] for i in tri]
print(f"Split: train={len(ti)} val={len(vi)} test={len(tei)}")
trl = [al[i] for i in ti]
cw = compute_class_weight('balanced', classes=np.unique(trl), y=trl)
cwd = {int(np.unique(trl)[i]): float(cw[i]) for i in range(len(np.unique(trl)))}

# %% 
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3), tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2), tf.keras.layers.RandomTranslation(0.1,0.1),
    tf.keras.layers.RandomBrightness(0.1), tf.keras.layers.RandomCrop(224,224),
])

def parse(fp, lbl):
    img = tf.io.read_file(fp); img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]); return img, lbl
def aug_train(img, lbl): return aug(img), lbl

Xt = [af[i] for i in ti]; yt = [al[i] for i in ti]
Xv = [af[i] for i in vi]; yv = [al[i] for i in vi]
Xte = [af[i] for i in tei]; yte = [al[i] for i in tei]

tds = tf.data.Dataset.from_tensor_slices((Xt, yt))
tds = tds.shuffle(2048).map(parse, tf.data.AUTOTUNE)
tds = tds.map(aug_train, tf.data.AUTOTUNE)
tds = tds.batch(BATCH_SIZE).repeat(10).prefetch(tf.data.AUTOTUNE)
vds = tf.data.Dataset.from_tensor_slices((Xv, yv))
vds = vds.map(parse, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
teds = tf.data.Dataset.from_tensor_slices((Xte, yte))
teds = teds.map(parse, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# %% 
MT = "__MODEL__"
if MT == "resnet50":
    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    pp = tf.keras.applications.resnet50.preprocess_input; ft = 143
elif MT == "efficientnetb0":
    base = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    pp = tf.keras.applications.efficientnet.preprocess_input; ft = 200
else:
    inp = tf.keras.Input((224,224,3))
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(2,activation='softmax')(x)
    model = tf.keras.Model(inp, out); model.summary(); base = None; ft = 0

if MT != "custom_cnn":
    inp = tf.keras.Input((224,224,3)); x = pp(inp); x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(2,activation='softmax')(x)
    model = tf.keras.Model(inp, out); model.summary()

if base: base.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h1 = model.fit(tds, validation_data=vds, epochs=20, class_weight=cwd, verbose=2)

if base:
    base.trainable = True
    for l in base.layers[:ft]: l.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h2 = model.fit(tds, validation_data=vds, epochs=30, class_weight=cwd, verbose=2)

tl, ta = model.evaluate(teds, verbose=2)
va = max(h1.history['val_accuracy'] + h2.history['val_accuracy'])
m = {"val_accuracy": float(va), "test_accuracy": float(ta), "num_classes": 2, "class_names": CLASS_NAMES}
with open("metrics.json","w") as f: json.dump(m, f)
os.makedirs("models", exist_ok=True)
model.save("models/grain___MODEL__.keras")
print(f"Test acc: {ta:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
yp = model.predict(teds); ypc = np.argmax(yp, axis=1)
print(classification_report(yte, ypc, target_names=CLASS_NAMES))
print(confusion_matrix(yte, ypc))
'''

KERNELS = [
    ("grain-resnet50-v5d", "Grain ResNet50 V5", "resnet50"),
    ("grain-efficientnet-v5d", "Grain EfficientNet V5", "efficientnetb0"),
    ("grain-customcnn-v5d", "Grain Custom CNN V5", "custom_cnn"),
]

for slug, title, model in KERNELS:
    print(f"\n--- {title} ---")
    nb = NB.replace("__TITLE__", title).replace("__MODEL__", model)
    cells = []
    for block in nb.split('# %%'):
        block = block.strip()
        if not block: continue
        ct = 'markdown' if block.startswith('[markdown]') else 'code'
        if ct == 'markdown': block = block[len('[markdown]'):].strip()
        cells.append({"cell_type": ct, "metadata": {}, "source": [block],
                     **({"outputs": [], "execution_count": None} if ct == 'code' else {})})
    
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python", "version": "3.10"}}, "cells": cells}
    
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, f"{slug}.ipynb"), 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        meta = {"id": f"juanestebanroja/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": False,
               "enable_gpu": True, "enable_internet": True, "dataset_sources": []}
        with open(os.path.join(tmpdir, "kernel-metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Pushing juanestebanroja/{slug} ...")
        api.kernels_push_cli(tmpdir, None, None)
        print(f"  https://www.kaggle.com/code/juanestebanroja/{slug}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

print("\nDone!")
