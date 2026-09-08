"""
Real accuracy benchmark against labeled synthetic data.

Unlike the old version of this script (which only checked "is the output
14 characters long" - a check that a wrong digit or a stray letter can
still pass), this compares extracted values against known-correct ground
truth from scripts/training/generate_trial_ids.py's manifest.json.

Usage:
    python scripts/training/generate_trial_ids.py --count 200   # if needed
    python tests/benchmark_accuracy.py --dataset data/synthetic_front
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from national_id_ocr.core.pipeline import Pipeline

logging.getLogger('national_id_ocr').setLevel(logging.ERROR)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def char_accuracy(expected: str, actual: str) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    dist = levenshtein(expected, actual or "")
    return max(0.0, 1.0 - dist / len(expected))


def run_benchmark(dataset_path: str, sample_size: int):
    manifest_path = Path(dataset_path) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest.json in {dataset_path}. Generate labeled data first:\n"
            f"  python scripts/training/generate_trial_ids.py --count {sample_size or 200} "
            f"--output {dataset_path}"
        )
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    if sample_size > 0:
        import random
        random.seed(42)
        manifest = random.sample(manifest, min(sample_size, len(manifest)))

    print(f"Dataset: {dataset_path}  |  Sampling: {len(manifest)} labeled images")
    print("-" * 60)

    pipeline = Pipeline()  # default engine (PaddleOCR) - see pipeline.py

    n = len(manifest)
    nid_exact = 0
    nid_checksum_valid = 0
    name_acc_sum = 0.0
    address_acc_sum = 0.0
    total_time_ms = 0.0
    errors = 0

    for entry in tqdm(manifest, desc="Benchmarking"):
        img_path = Path(dataset_path) / entry["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            errors += 1
            continue

        gt = entry["data"]
        start = time.time()
        try:
            result = pipeline.process_image(image)
        except Exception:
            errors += 1
            continue
        total_time_ms += (time.time() - start) * 1000

        front = result.front
        extracted_nid = front.national_id if front else None

        if extracted_nid == gt["national_id"]:
            nid_exact += 1
        if result.decoded is not None:
            nid_checksum_valid += 1

        name_acc_sum += char_accuracy(gt["full_name"], front.full_name if front else "")
        address_acc_sum += char_accuracy(gt["address"], front.address if front else "")

    print("-" * 60)
    print("Results (against ground truth, not just shape checks):")
    print(f"  National ID exact match:     {nid_exact}/{n}  ({nid_exact/n*100:.1f}%)")
    print(f"  National ID checksum valid:  {nid_checksum_valid}/{n}  ({nid_checksum_valid/n*100:.1f}%)")
    print(f"  Full name char accuracy:     {name_acc_sum/n*100:.1f}%  (avg, Levenshtein-based)")
    print(f"  Address char accuracy:       {address_acc_sum/n*100:.1f}%  (avg, Levenshtein-based)")
    print(f"  Avg latency:                 {total_time_ms/n:.1f} ms")
    print(f"  Errors:                      {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_front")
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.sample)
