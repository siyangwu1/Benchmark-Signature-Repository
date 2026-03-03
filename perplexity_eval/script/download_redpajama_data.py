"""
Script to download RedPajama-Data-1T-Sample dataset files from Hugging Face.
This script fetches the dataset information and downloads all files to the
directory specified by paths.redpajama_download_dir in config/config.yaml.
"""

import requests
import os
from pathlib import Path
from urllib.parse import urlparse
import time
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    """Walk up from this file until config/config.yaml is found."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Cannot locate config/config.yaml")


def _load_config() -> tuple[dict, Path]:
    repo_root = _find_repo_root()
    with open(repo_root / "config" / "config.yaml") as f:
        return yaml.safe_load(f), repo_root


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


# ── Download logic ────────────────────────────────────────────────────────────

def fetch_dataset_info():
    """Fetch dataset information from Hugging Face API."""
    url = "https://huggingface.co/api/datasets/togethercomputer/RedPajama-Data-1T-Sample/parquet/plain_text/train"

    print("Fetching dataset information from Hugging Face...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset info: {e}")
        return None


def parse_file_info(dataset_info):
    """Parse the dataset info to extract file URLs and names."""
    files = []

    if not dataset_info:
        return files

    if isinstance(dataset_info, list):
        for url in dataset_info:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            files.append({"url": url, "filename": filename})
    else:
        print(f"Unexpected data format: {type(dataset_info)}")
        print("Expected a list of URLs")

    return files


def download_file(url, local_path):
    """Download a single file from URL to local path."""
    try:
        print(f"Downloading: {os.path.basename(local_path)}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Downloaded: {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(local_path)}: {e}")
        return False


def main():
    cfg, repo_root = _load_config()

    # Apply HF_TOKEN from config if not already set in the shell
    _hf_token = str(cfg.get("environment", {}).get("hf_token", "")).strip()
    if _hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _hf_token

    data_dir = _resolve(cfg["paths"]["redpajama_download_dir"], repo_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = fetch_dataset_info()

    if not dataset_info:
        print("Failed to fetch dataset information. Exiting.")
        return

    print(f"\nDataset information received:")
    print(f"Found {len(dataset_info)} files")

    files = parse_file_info(dataset_info)

    if not files:
        print("\nNo files found in the dataset info. Exiting.")
        return

    print(f"\nFiles to download:")
    for file_info in files:
        print(f"  - {file_info['filename']}")

    print(f"\nStarting download to {data_dir.absolute()}...")
    successful_downloads = 0

    for file_info in files:
        filename = file_info["filename"]
        file_url = file_info["url"]
        local_path = data_dir / filename

        if local_path.exists():
            print(f"File already exists, skipping: {filename}")
            successful_downloads += 1
            continue

        if download_file(file_url, local_path):
            successful_downloads += 1

        time.sleep(0.5)

    print(f"\nDownload complete! {successful_downloads}/{len(files)} files downloaded successfully.")
    print(f"Files saved to: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
