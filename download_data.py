"""
download_data.py — Pulls the required Kaggle datasets locally.

Prerequisite: kaggle.json must already be placed at
  Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
  macOS/Linux: ~/.kaggle/kaggle.json

Usage:
    python download_data.py
"""

import os
import subprocess
import sys

import config


def check_kaggle_credentials():
    """Verify kaggle.json exists before attempting any downloads."""
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("✗ kaggle.json not found at:", kaggle_json)
        print("\nFix:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under the API section")
        print(f"  3. Move the downloaded file to: {kaggle_json}")
        sys.exit(1)

    print(f"✓ Found kaggle.json at {kaggle_json}")


def download_dataset(dataset_slug: str, dest_dir: str):
    """Run the kaggle CLI to download and unzip a dataset."""
    print(f"\nDownloading {dataset_slug} ...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", dest_dir,
            "--unzip",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"✗ Failed to download {dataset_slug}")
        print(result.stderr)
        if "403" in result.stderr:
            print("\nThis is usually a terms-of-use issue. Fix:")
            print(f"  Visit https://www.kaggle.com/datasets/{dataset_slug}")
            print("  in your browser and click 'Download' once to accept the terms.")
        sys.exit(1)
    print(result.stdout)
    print(f"✓ Downloaded and extracted {dataset_slug}")


def main():
    check_kaggle_credentials()
    config.ensure_dirs()

    download_dataset("najzeko/steam-reviews-2021", config.DATA_DIR)
    download_dataset("nikdavis/steam-store-games", config.DATA_DIR)

    print("\nFiles in data directory:")
    for f in os.listdir(config.DATA_DIR):
        path = os.path.join(config.DATA_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
