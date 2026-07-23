"""Audit image provenance, duplication, and split isolation."""
import argparse
import hashlib
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED = ["image", "source_image_id", "patient_id", "lesion_id"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path):
    image = Image.open(path).convert("L").resize((9, 8))
    values = np.asarray(image)
    bits = values[:, 1:] > values[:, :-1]
    return sum(int(bit) << i for i, bit in enumerate(bits.ravel()))


def hamming(left, right):
    return (left ^ right).bit_count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="src/data/split_manifest.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--near-distance", type=int, default=4)
    parser.add_argument("--out", default="src/data/dataset_audit.json")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    missing = [column for column in REQUIRED if column not in manifest.columns]
    if missing:
        raise ValueError("The source manifest is missing: " + ", ".join(missing))
    if manifest[REQUIRED].isna().any().any():
        raise ValueError("The source identifiers, patient IDs, lesion IDs, and splits must be complete.")

    records = []
    for row in manifest.itertuples(index=False):
        path = os.path.join(args.images, row.image)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        records.append({
            "image": row.image,
            "sha256": sha256(path),
            "dhash": dhash(path),
        })

    exact = {}
    for record in records:
        exact.setdefault(record["sha256"], []).append(record["image"])
    exact_groups = [images for images in exact.values() if len(images) > 1]

    near_pairs = []
    for left, right in combinations(records, 2):
        distance = hamming(left["dhash"], right["dhash"])
        if distance <= args.near_distance:
            near_pairs.append({
                "left": left["image"], "right": right["image"],
                "distance": distance,
            })

    leakage = {"patient_id": [], "lesion_id": []}
    split_checksums = {}
    if "split" in manifest.columns:
        for column in ("patient_id", "lesion_id"):
            counts = manifest.groupby(column).split.nunique()
            leakage[column] = counts[counts > 1].index.astype(str).tolist()
        for split, group in manifest.groupby("split"):
            payload = "\n".join(sorted(group.image.astype(str))) + "\n"
            split_checksums[split] = hashlib.sha256(payload.encode()).hexdigest()

    report = {
        "images": int(len(manifest)),
        "patients": int(manifest.patient_id.nunique()),
        "lesions": int(manifest.lesion_id.nunique()),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_pairs,
        "patient_split_leakage": leakage["patient_id"],
        "lesion_split_leakage": leakage["lesion_id"],
        "split_checksums": split_checksums,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)

    failed = exact_groups or near_pairs or leakage["patient_id"] or leakage["lesion_id"]
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit("Dataset audit failed.")


if __name__ == "__main__":
    main()
