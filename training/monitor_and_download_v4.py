"""
V4 Binary Models — Auto-download watcher.

Polls Kaggle kernel status every 5 minutes. When a kernel completes
successfully (status="complete"), downloads its output to
training/models/v4/. When all 6 are downloaded, builds the ensembles.

Usage:
    python monitor_and_download_v4.py

Leave this running in a terminal. Press Ctrl+C to stop.
"""
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\elkaw\Desktop\CoffeeVision AI")
MODELS_DIR = BASE_DIR / "training" / "models" / "v4"
POLL_INTERVAL = 300  # 5 minutes

TOKENS = {
    "julianmedinamonje45": "KGAT_5e205dcd6f68c3b4ff4c4c121a149bf8",
    "vann234": "KGAT_a2b7c5cc854eea821166afe9cc1fa585",
    "juanveru": "KGAT_fcb1ba0670cfb603b1a1ab6053eb7961",
}

KERNELS = [
    ("julianmedinamonje45", "coffeevision-leaf-resnet50-v4", "leaf_resnet50"),
    ("julianmedinamonje45", "coffeevision-leaf-customcnn-v4", "leaf_custom_cnn"),
    ("vann234", "coffeevision-grain-resnet50-v4", "grain_resnet50"),
    ("vann234", "coffeevision-grain-customcnn-v4", "grain_custom_cnn"),
    ("juanveru", "coffeevision-leaf-effnet-v4", "leaf_efficientnetb0"),
    ("juanveru", "coffeevision-grain-effnet-v4", "grain_efficientnetb0"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api(user: str):
    """Create an authenticated Kaggle API client for the given user."""
    token = TOKENS[user]
    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def check_status(api, user: str, slug: str) -> str:
    """Return kernel status string: 'running', 'complete', 'error', 'queued', etc."""
    try:
        result = api.kernels_status(kernel=f"{user}/{slug}")
        # The Kaggle API returns an object with .status attribute (enum)
        raw = getattr(result, "status", None)
        if raw is None:
            return "unknown"
        # Convert enum to string
        status_str = str(raw).lower()
        # Clean up KernelWorkerStatus.RUNNING -> running
        if "running" in status_str:
            return "running"
        if "complete" in status_str:
            return "complete"
        if "error" in status_str:
            return "error"
        if "queued" in status_str:
            return "queued"
        if "cancel" in status_str:
            return "cancelled"
        return status_str
    except Exception as e:
        err = str(e)
        if "404" in err or "Not Found" in err:
            return "not_found"
        if "Permission" in err or "denied" in err:
            return "no_access"
        return f"api_error: {err[:60]}"


def download_output(api, user: str, slug: str, label: str):
    """Download kernel output to models/v4/{label}/"""
    dest = MODELS_DIR / label
    dest.mkdir(parents=True, exist_ok=True)

    try:
        # Use kernels_output to list files, then download
        files = api.kernels_output(f"{user}/{slug}")
        
        # kernels_output returns a list of file dicts or a response object
        if hasattr(files, 'files'):
            file_list = files.files
        elif isinstance(files, list):
            file_list = files
        else:
            # Try reading as dict
            file_list = files if isinstance(files, dict) else []

        downloaded = 0
        for f in file_list:
            fname = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            if fname and (fname.endswith(".keras") or fname.endswith(".json") or fname.endswith(".csv") or fname.endswith(".txt")):
                # Download individual file
                api.kernels_pull(f"{user}/{slug}", path=str(dest), file_name=fname)
                downloaded += 1
                print(f"    Downloaded: {fname}")

        if downloaded == 0:
            # Fallback: try pulling everything
            print(f"    Pulling all output files...")
            api.kernels_pull(f"{user}/{slug}", path=str(dest))
            # List what we got
            for f in dest.iterdir():
                if f.is_file():
                    print(f"    Downloaded: {f.name}")
                    downloaded += 1

        if downloaded > 0:
            print(f"  [OK] Downloaded {downloaded} files to {dest}")
            return True
        else:
            print(f"  [WARN] No output files found for {label}")
            return False

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CoffeeVision V4 — Auto Monitor & Download")
    print(f"Polling every {POLL_INTERVAL // 60} minutes")
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = set()
    failed = set()

    while len(downloaded) + len(failed) < len(KERNELS):
        print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
        
        for user, slug, label in KERNELS:
            if label in downloaded or label in failed:
                continue

            api = get_api(user)
            status = check_status(api, user, slug)
            print(f"  {label:25s} [{user}/{slug[:20]}] → {status.upper()}")
            sys.stdout.flush()

            if status == "complete":
                print(f"  >>> {label} COMPLETED! Downloading...")
                ok = download_output(api, user, slug, label)
                if ok:
                    downloaded.add(label)
                else:
                    failed.add(label)
            elif status in ("error", "cancelled", "not_found"):
                print(f"  [FAIL] {label} ended with status '{status}'. Will not retry.")
                failed.add(label)

        # Show progress
        remaining = len(KERNELS) - len(downloaded) - len(failed)
        if remaining > 0:
            print(f"\n  Progress: {len(downloaded)} done, {len(failed)} failed, {remaining} remaining")
            print(f"  Next check in {POLL_INTERVAL // 60} minutes...")
            time.sleep(POLL_INTERVAL)

    # All done (or all accounted for)
    print("\n" + "=" * 60)
    print("ALL KERNELS PROCESSED")
    print("=" * 60)
    print(f"  Downloaded: {len(downloaded)}/6")
    if downloaded:
        for label in sorted(downloaded):
            print(f"    ✅ {label}")
    if failed:
        print(f"  Failed: {len(failed)}/6")
        for label in sorted(failed):
            print(f"    ❌ {label}")

    if len(downloaded) >= 4:
        print("\n[INFO] Enough models downloaded. Consider building ensembles manually:")
        print(f"  cd {BASE_DIR / 'training'}")
        print(f"  python build_ensemble.py --dataset leaf --num-classes 2 --models-dir {MODELS_DIR}")
        print(f"  python build_ensemble.py --dataset grain --num-classes 2 --models-dir {MODELS_DIR}")
    elif len(downloaded) > 0:
        print(f"\n[WARN] Only {len(downloaded)} models downloaded. May need to retry failed ones.")
    else:
        print("\n[ERROR] No models downloaded. Check Kaggle kernels manually.")
        print("  https://www.kaggle.com/julianmedinamonje45/kernels")
        print("  https://www.kaggle.com/vann234/kernels")
        print("  https://www.kaggle.com/juanveru/kernels")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
