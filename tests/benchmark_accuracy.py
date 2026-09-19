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

from egyptian_national_id_ocr.core.pipeline import Pipeline

from egyptian_national_id_ocr.postprocessing.text_utils import char_accuracy
from egyptian_national_id_ocr.postprocessing.enum_matcher import match_enum, RELIGION_VALUES, MARITAL_STATUS_VALUES

logging.getLogger('egyptian_national_id_ocr').setLevel(logging.ERROR)


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
    serial_exact = 0
    total_time_ms = 0.0
    errors = 0

    has_back = all("back_filename" in entry for entry in manifest)
    back_nid_exact = 0
    back_issue_date_exact = 0
    back_expiry_date_exact = 0
    back_profession_acc_sum = 0.0
    back_gender_correct = 0
    back_religion_correct = 0
    back_marital_correct = 0
    back_n = 0
    back_errors = 0

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
        if front is not None and front.card_serial_number == gt["serial_number"]:
            serial_exact += 1

        if has_back:
            back_n += 1
            back_img_path = Path(dataset_path) / entry["back_filename"]
            back_image = cv2.imread(str(back_img_path))
            if back_image is None:
                back_errors += 1
                continue
            try:
                back_result = pipeline.process_image(back_image)
            except Exception:
                back_errors += 1
                continue

            back = back_result.back
            if back is None:
                continue

            if back.national_id == gt["national_id"]:
                back_nid_exact += 1
            # issue_date/expiry_date come back as bare digit strings (the
            # digit classifier has no separator class - see
            # DigitClassifierEngine) while ground truth includes "/" -
            # strip separators from both sides for a fair comparison.
            if (back.issue_date or "") == gt["issue_date"].replace("/", ""):
                back_issue_date_exact += 1
            if (back.expiry_date or "") == gt["expiry_date"].replace("/", ""):
                back_expiry_date_exact += 1
            back_profession_acc_sum += char_accuracy(gt["profession"], back.profession or "")
            if back.gender is not None and back.gender.value == (
                "male" if gt["gender"] == "ذكر" else "female"
            ):
                back_gender_correct += 1
            expected_religion, _ = match_enum(gt["religion"], RELIGION_VALUES)
            if back.religion == expected_religion:
                back_religion_correct += 1
            expected_marital, _ = match_enum(gt["marital_status"], MARITAL_STATUS_VALUES)
            if back.marital_status == expected_marital:
                back_marital_correct += 1

    print("-" * 60)
    print("Results (against ground truth, not just shape checks):")
    print(f"  National ID exact match:     {nid_exact}/{n}  ({nid_exact/n*100:.1f}%)")
    print(f"  National ID checksum valid:  {nid_checksum_valid}/{n}  ({nid_checksum_valid/n*100:.1f}%)")
    print(f"  Full name char accuracy:     {name_acc_sum/n*100:.1f}%  (avg, Levenshtein-based)")
    print(f"  Address char accuracy:       {address_acc_sum/n*100:.1f}%  (avg, Levenshtein-based)")
    print(f"  Serial number exact match:   {serial_exact}/{n}  ({serial_exact/n*100:.1f}%)")
    print(f"  Avg latency:                 {total_time_ms/n:.1f} ms")
    print(f"  Errors:                      {errors}")

    if has_back and back_n:
        print("-" * 60)
        print("Back side:")
        print(f"  National ID exact match:     {back_nid_exact}/{back_n}  ({back_nid_exact/back_n*100:.1f}%)")
        print(f"  Issue date exact match:      {back_issue_date_exact}/{back_n}  ({back_issue_date_exact/back_n*100:.1f}%)")
        print(f"  Expiry date exact match:     {back_expiry_date_exact}/{back_n}  ({back_expiry_date_exact/back_n*100:.1f}%)")
        print(f"  Profession char accuracy:    {back_profession_acc_sum/back_n*100:.1f}%  (avg, Levenshtein-based)")
        print(f"  Gender correct:              {back_gender_correct}/{back_n}  ({back_gender_correct/back_n*100:.1f}%)")
        print(f"  Religion correct:            {back_religion_correct}/{back_n}  ({back_religion_correct/back_n*100:.1f}%)")
        print(f"  Marital status correct:      {back_marital_correct}/{back_n}  ({back_marital_correct/back_n*100:.1f}%)")
        print(f"  Errors:                      {back_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_front")
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.sample)
