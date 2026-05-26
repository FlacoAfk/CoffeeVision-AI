"""Discover the actual Kaggle dataset mount path by listing input dirs."""
import os
import json
import tempfile
import shutil

KAGGLE_TOKENS = {
    "julianmedinamonje45": "KGAT_026351ca8f39d7e6bf42e0d57172c31e",
    "juanveru": "KGAT_0d5689dd900f4144054873d405086e7e",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
}

# We'll push a diagnostic kernel that just lists /kaggle/input/
def make_diagnostic_notebook():
    cell1 = """import os
print("=== /kaggle/input/ contents ===")
for item in sorted(os.listdir('/kaggle/input/')):
    full = os.path.join('/kaggle/input/', item)
    if os.path.isdir(full):
        subitems = sorted(os.listdir(full))[:20]
        print(f"  {item}/ ({len(os.listdir(full))} items)")
        for s in subitems:
            subfull = os.path.join(full, s)
            if os.path.isdir(subfull):
                print(f"    {s}/ ({len(os.listdir(subfull))} items)")
            else:
                print(f"    {s}")
    else:
        print(f"  {item}")
print("\\n=== Done ===")
"""
    cells = [{
        "cell_type": "code",
        "metadata": {},
        "source": [cell1 + "\n"],
        "outputs": [],
        "execution_count": None,
    }]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "cells": cells,
    }


def push_diagnostic(token, username, slug, title, dataset_slug):
    os.environ["KAGGLE_API_TOKEN"] = token
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
        os.environ.pop(k, None)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    tmp = tempfile.mkdtemp(prefix="diag_")
    try:
        nb_path = os.path.join(tmp, "notebook.ipynb")
        with open(nb_path, "w") as f:
            json.dump(make_diagnostic_notebook(), f)

        metadata = {
            "id": f"{username}/{slug}",
            "title": title,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_internet": "true",
            "dataset_sources": [dataset_slug],
        }
        meta_path = os.path.join(tmp, "kernel-metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Pushing {slug}...", end=" ", flush=True)
        api.kernels_push(tmp)
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Push diagnostics for each account's dataset
DIAGS = [
    ("julianmedinamonje45", "coffevision-leaf-cls", "julianmedinamonje45/coffevision-leaf-cls"),
    ("vann234", "coffevision-grain-cls", "vann234/coffevision-grain-cls"),
    ("juanveru", "coffevision-leaf-cls-juanveru", "juanveru/coffevision-leaf-cls"),
]

print("Pushing diagnostic kernels...")
for username, diag_slug, dataset_slug in DIAGS:
    push_diagnostic(
        token=KAGGLE_TOKENS[username],
        username=username,
        slug=diag_slug,
        title=f"Diagnostic Path {diag_slug}",
        dataset_slug=dataset_slug,
    )

print("\nDone! Wait ~2 min then run check_diag.py")
