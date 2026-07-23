"""
Freeze the dataset into train/val/test splits and record checksums.

The input metadata must contain source image, patient, and lesion identifiers.
Patients are assigned to one split only. Indeterminate images remain in the
manifest but are excluded from the binary endpoint.
"""
import os, csv, json, hashlib, random
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # llava_skin_cancer/
IMG_DIR = os.path.join(ROOT, "images")
META = os.path.join(ROOT, "metadata.csv")
JSONL = os.path.join(ROOT, "data.jsonl")
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

SEED = 42
random.seed(SEED)

def binary_label(dx):
    dx = str(dx).strip()
    if dx == "Malignant":
        return 1
    if dx == "Benign":
        return 0
    return -1  # Indeterminate / unknown -> excluded from classification

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    df = pd.read_csv(META)
    required = {"source_image_id", "patient_id", "lesion_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Source-linked splitting requires these metadata columns: " + ", ".join(missing)
        )
    df = df[df["isic_id"].apply(lambda x: str(x).strip().isdigit())].copy()
    df["isic_id"] = df["isic_id"].astype(int)
    df["image"] = df["isic_id"].apply(lambda i: f"{i:03d}.jpg")
    df["y"] = df["diagnosis_1"].apply(binary_label)
    df["confirm"] = df["diagnosis_confirm_type"].fillna("not_reported")

    # keep only images that exist on disk
    df = df[df["image"].apply(lambda im: os.path.exists(os.path.join(IMG_DIR, im)))].copy()

    # Patient-level split, stratified by the highest-risk diagnosis recorded for each patient.
    risk_order = {"Benign": 0, "Indeterminate": 1, "Malignant": 2}
    patient_label = {}
    for patient, group in df.groupby("patient_id"):
        patient_label[patient] = max(
            group["diagnosis_1"], key=lambda value: risk_order.get(value, -1)
        )
    rows_by_cls = {}
    for patient, label in patient_label.items():
        rows_by_cls.setdefault(label, []).append(patient)
    split_of = {}
    for cls, patients in rows_by_cls.items():
        patients = sorted(patients)
        random.Random(SEED).shuffle(patients)
        n = len(patients); n_tr = int(round(0.60 * n)); n_va = int(round(0.15 * n))
        for i, patient in enumerate(patients):
            split_of[patient] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    df["split"] = df["patient_id"].map(split_of)

    manifest = df[[
        "isic_id", "source_image_id", "patient_id", "lesion_id", "image",
        "diagnosis_1", "y", "confirm", "split", "melanocytic",
    ]]
    manifest = manifest.sort_values("isic_id")
    man_path = os.path.join(OUT, "split_manifest.csv")
    manifest.to_csv(man_path, index=False)

    # Q&A split files: assign each QA pair to its image's split
    img2split = dict(zip(df["image"], df["split"]))
    qa = {"train": [], "val": [], "test": []}
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            im = os.path.basename(d["image"])
            s = img2split.get(im)
            if s:
                qa[s].append(d)
    for s in qa:
        p = os.path.join(OUT, f"qa_{s}.jsonl")
        with open(p, "w") as f:
            for d in qa[s]:
                f.write(json.dumps(d) + "\n")

    # Checksums for every frozen split file
    files = [man_path] + [os.path.join(OUT, f"qa_{s}.jsonl") for s in ("train", "val", "test")]
    with open(os.path.join(OUT, "checksums.sha256"), "w") as f:
        for p in files:
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")

    # Summary
    summ = {
        "seed": SEED,
        "n_images": int(len(df)),
        "split_images": manifest["split"].value_counts().to_dict(),
        "binary_counts_overall": {
            "benign": int((df["y"] == 0).sum()),
            "malignant": int((df["y"] == 1).sum()),
            "indeterminate_excluded": int((df["y"] == -1).sum()),
        },
        "qa_counts": {s: len(qa[s]) for s in qa},
        "confirm_types": df["confirm"].value_counts().to_dict(),
        "unique_patients": int(df["patient_id"].nunique()),
        "unique_lesions": int(df["lesion_id"].nunique()),
        "note": "Patient-level stratified split.",
    }
    with open(os.path.join(OUT, "split_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
