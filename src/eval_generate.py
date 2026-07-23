"""
Generation evaluation for a trained DermaAttr-VLM config: BLEU-4, ROUGE-L, METEOR,
reported overall and per question-category. Runs on a fixed
stratified subset of the frozen test Q&A to bound GPU time; subset is seeded and saved.

Usage: python eval_generate.py --model models/full_s0 --config full --n 300
"""
import os, sys, json, argparse, random, re
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_derma as T
from derma_attr_vlm import D_ATTR
from eval_classify import load_trained

IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    try: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    except Exception: pass
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
smooth = SmoothingFunction().method1


def rouge_l(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r or not h:
        return 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[len(r)][len(h)]
    prec = lcs / len(h); rec = lcs / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def categorize(q):
    q = q.lower()
    if any(k in q for k in ["diagnos", "malign", "benign", "risk"]): return "diagnosis"
    if any(k in q for k in ["pattern", "network", "globul", "streak", "veil"]): return "pattern_description"
    if any(k in q for k in ["asymmetr", "border", "color", "colour", "abcd"]): return "attribute_description"
    if any(k in q for k in ["feature", "detect"]): return "feature_recognition"
    if any(k in q for k in ["why", "reason", "because", "suggest", "consistent"]): return "clinical_reasoning"
    return "observation"


def management_question(q):
    q = q.lower()
    return any(k in q for k in [
        "treat", "manage", "biopsy", "excis", "refer", "follow-up",
        "follow up", "surgery", "therapy", "medication",
    ])


def stratified_items(items, n, seed=42):
    """Select equal numbers from the five prespecified question categories."""
    import random
    categories = [
        "diagnosis", "feature_recognition", "attribute_description",
        "pattern_description", "clinical_reasoning",
    ]
    groups = {category: [] for category in categories}
    for item in items:
        q = item["conversations"][0]["value"].replace("<image>", "").strip()
        category = categorize(q)
        if category in groups and not management_question(q):
            groups[category].append(item)
    rng = random.Random(seed)
    target = n // len(categories)
    selected = []
    for category in categories:
        rng.shuffle(groups[category])
        selected.extend(groups[category][:target])
    if len(selected) != target * len(categories):
        raise ValueError("The test set does not contain enough items in every category.")
    rng.shuffle(selected)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    name = os.path.basename(args.model.rstrip("/"))

    items = []
    with open(os.path.join(DATA, "qa_test.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line: items.append(json.loads(line))
    items = stratified_items(items, args.n, seed=42)

    attrs = T.load_attrs()
    images = [os.path.basename(d["image"]) for d in items]
    control = T.shuffled_attr_map(images, attrs, args.seed) if args.config == "shuffled" else attrs
    model, proc, tok = load_trained(args.model, args.config, args.seed)
    prefix_ids = torch.tensor(tok("USER: ", add_special_tokens=True).input_ids)

    rows = []
    for d in items:
        im = os.path.basename(d["image"])
        img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
        px = proc.image_processor(img, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        a = torch.tensor(attrs.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        a_control = torch.tensor(control.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        q = d["conversations"][0]["value"].replace("<image>", "").strip()
        ref = d["conversations"][1]["value"].strip()
        mid = tok(q + "\nASSISTANT: ", add_special_tokens=False).input_ids[:160]
        mid_t = torch.tensor(mid).unsqueeze(0)
        pred = model.generate(
            px, a, prefix_ids, mid_t, len(mid), max_new_tokens=96,
            a_control=a_control
        )
        pred = pred.strip()
        cat = categorize(q)
        rt, ht = ref.lower().split(), pred.lower().split()
        b4 = sentence_bleu([rt], ht, weights=(.25, .25, .25, .25), smoothing_function=smooth) if ht else 0.0
        try: met = meteor_score([rt], ht)
        except Exception: met = 0.0
        rl = rouge_l(ref.lower(), pred.lower())
        rows.append(dict(image=im, category=cat, question=q, reference=ref, prediction=pred,
                         bleu4=b4, rougeL=rl, meteor=met))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"gen_{name}.csv"), index=False)
    overall = dict(model=name, n=len(df), bleu4=df.bleu4.mean(),
                   rougeL=df.rougeL.mean(), meteor=df.meteor.mean())
    cats = df.groupby("category")[["bleu4", "rougeL", "meteor"]].mean().reset_index()
    cats.to_csv(os.path.join(OUT, f"gen_{name}_bycat.csv"), index=False)
    with open(os.path.join(OUT, f"gen_{name}_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))
    print(cats.to_string(index=False))


if __name__ == "__main__":
    main()
