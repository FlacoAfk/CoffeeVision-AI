"""Push deeper diagnostic that lists 3 levels deep."""
import os, json, tempfile, shutil

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

def make_diag_nb():
    cell = """import os

def list_deep(path, depth=0, max_depth=4):
    indent = "  " * depth
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(f"{indent}[permission denied]")
        return
    for item in items[:30]:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            n_items = 0
            try:
                n_items = len(os.listdir(full))
            except:
                pass
            print(f"{indent}{item}/ ({n_items} items)")
            if depth < max_depth:
                list_deep(full, depth + 1, max_depth)
        else:
            print(f"{indent}{item}")

print("=== /kaggle/input/ tree ===")
list_deep("/kaggle/input/")
"""
    cells = [{"cell_type": "code", "metadata": {}, "source": [cell + "\n"], "outputs": [], "execution_count": None}]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10.12"}},
            "cells": cells}

def push(token, username, slug, title, dataset_slug):
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]: os.environ.pop(k, None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "notebook.ipynb"), "w") as f: json.dump(make_diag_nb(), f)
        meta = {"id": f"{username}/{slug}", "title": title, "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": "true",
                "enable_gpu": "false", "enable_internet": "true", "dataset_sources": [dataset_slug]}
        with open(os.path.join(tmp, "kernel-metadata.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"  Pushing {slug}...", end=" ", flush=True)
        api.kernels_push(tmp)
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

DIAGS = [
    ("julianmedinamonje45", "diag-leaf-v4", "Diagnostic Leaf V4", "julianmedinamonje45/coffevision-leaf-cls"),
    ("vann234", "diag-grain-v4", "Diagnostic Grain V4", "vann234/coffevision-grain-cls"),
    ("juanveru", "diag-leaf-juanveru-v4", "Diagnostic Leaf Juanveru V4", "juanveru/coffevision-leaf-cls"),
]

for u, s, t, ds in DIAGS:
    push(KAGGLE_TOKENS[u], u, s, t, ds)
print("Done! Wait ~60s then download logs")
