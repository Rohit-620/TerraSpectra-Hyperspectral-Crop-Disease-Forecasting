"""
TerraSpectra - Hyperspectral Dataset Acquisition and Verification Script
Day 1: Automated acquisition and integrity validation of agricultural hyperspectral benchmarks.
"""

import os
import sys
import urllib.request
import ssl
import numpy as np
import scipy.io as sio

DATASETS = {
    "indian_pines_corrected": {
        "filename": "Indian_pines_corrected.mat",
        "urls": [
            "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        ],
        "key": "indian_pines_corrected",
        "expected_shape": (145, 145, 200),
        "description": "AVIRIS Indian Pines corrected reflectance cube (200 bands, water absorption bands removed)"
    },
    "indian_pines_gt": {
        "filename": "Indian_pines_gt.mat",
        "urls": [
            "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
        ],
        "key": "indian_pines_gt",
        "expected_shape": (145, 145),
        "description": "Indian Pines ground truth labels (16 agricultural/vegetation classes)"
    },
    "indian_pines_raw": {
        "filename": "Indian_pines.mat",
        "urls": [
            "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines.mat",
            "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
        ],
        "key": "indian_pines",
        "expected_shape": (145, 145, 220),
        "description": "AVIRIS Indian Pines raw radiance/reflectance cube (all 220 calibrated bands including atmospheric noise)"
    },
    "salinas_corrected": {
        "filename": "Salinas_corrected.mat",
        "urls": [
            "https://huggingface.co/datasets/danaroth/salinas/resolve/main/Salinas_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
        ],
        "key": "salinas_corrected",
        "expected_shape": (512, 217, 204),
        "description": "AVIRIS Salinas agricultural scene (204 bands, 512x217 pixels, high spatial resolution 3.7m)"
    },
    "salinas_gt": {
        "filename": "Salinas_gt.mat",
        "urls": [
            "https://huggingface.co/datasets/danaroth/salinas/resolve/main/Salinas_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/f/fb/Salinas_gt.mat"
        ],
        "key": "salinas_gt",
        "expected_shape": (512, 217),
        "description": "Salinas ground truth labels (16 agricultural crop/phenology classes)"
    }
}

def download_file(urls, destination_path):
    """Download a file with retry and mirror fallback."""
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 10000:
        print(f"  [ALREADY EXISTS] {os.path.basename(destination_path)} ({os.path.getsize(destination_path) / 1024 / 1024:.2f} MB)", flush=True)
        return True

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    for url in urls:
        print(f"  Attempting download from: {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 128

                with open(destination_path, 'wb') as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r    Progress: {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB ({percent:.1f}%)", end='', flush=True)
                        else:
                            print(f"\r    Downloaded: {downloaded / 1024 / 1024:.2f} MB", end='', flush=True)
                print(flush=True)
                print(f"  [SUCCESS] Successfully downloaded to {destination_path} ({os.path.getsize(destination_path)/1024/1024:.2f} MB)", flush=True)
                return True
        except Exception as e:
            print(f"\n  [FAILED] Mirror failed: {e}. Trying next mirror...", flush=True)
            if os.path.exists(destination_path):
                os.remove(destination_path)

    return False

def verify_dataset(filepath, expected_key, expected_shape):
    """Verify that the .mat file loads correctly and has expected dimensions."""
    try:
        mat_dict = sio.loadmat(filepath)
        found_key = None
        for k in mat_dict.keys():
            if not k.startswith('__'):
                if k.lower() == expected_key.lower():
                    found_key = k
                    break
        if not found_key:
            keys = [k for k in mat_dict.keys() if not k.startswith('__')]
            if keys:
                found_key = keys[0]

        if not found_key:
            return False, f"No data matrix found in {filepath}. Keys: {list(mat_dict.keys())}"

        data = mat_dict[found_key]
        if data.shape != expected_shape:
            if sorted(data.shape) == sorted(expected_shape):
                return True, f"Valid (transposed): key='{found_key}', shape={data.shape}, dtype={data.dtype}"
            return False, f"Shape mismatch: expected {expected_shape}, got {data.shape}"

        return True, f"Verified valid: key='{found_key}', shape={data.shape}, dtype={data.dtype}, min={np.min(data)}, max={np.max(data)}"
    except Exception as e:
        return False, f"Corrupted or invalid MAT file: {e}"

def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(target_dir, exist_ok=True)
    print("=" * 75, flush=True)
    print(" TerraSpectra: Day 1 Hyperspectral Dataset Acquisition & Verification", flush=True)
    print(f" Destination Directory: {target_dir}", flush=True)
    print("=" * 75, flush=True)

    summary = []
    all_success = True

    for name, meta in DATASETS.items():
        print(f"\nProcessing [{name}]: {meta['description']}", flush=True)
        dest = os.path.join(target_dir, meta["filename"])
        success = download_file(meta["urls"], dest)
        if success:
            valid, msg = verify_dataset(dest, meta["key"], meta["expected_shape"])
            print(f"  Verification: {msg}", flush=True)
            summary.append((name, meta["filename"], "PASS" if valid else "FAIL", msg))
            if not valid:
                all_success = False
        else:
            print(f"  [ERROR] Could not download {meta['filename']} from any mirror.", flush=True)
            summary.append((name, meta["filename"], "DOWNLOAD_FAILED", "All mirrors unreachable"))
            all_success = False

    print("\n" + "=" * 75, flush=True)
    print(" ACQUISITION SUMMARY REPORT", flush=True)
    print("=" * 75, flush=True)
    for name, fname, status, details in summary:
        print(f"[{status:15s}] {fname:28s} -> {details}", flush=True)

    if all_success:
        print("\nAll target hyperspectral benchmark datasets acquired and verified successfully!", flush=True)
    else:
        print("\nSome datasets could not be downloaded or verified. Please check connectivity.", flush=True)

if __name__ == "__main__":
    main()
