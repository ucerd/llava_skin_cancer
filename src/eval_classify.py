"""
Evaluate a trained DermaAttr-VLM config's classification head on the frozen test set.
Produces case-level scores (one per unique image, indeterminate excluded) and a
validation-selected threshold, consumed by stats.py.

Usage: python eval_classify.py --model models/full_s0 --config full
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_derma as T
from derma_attr_vlm import ATTRS, D_ATTR
from peft import set_peft_model_state_dict

IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")


def load_trained(model_dir, config, seed=0):
    cfg = T.CONFIGS[config]
    model, proc, tok = T.build_model(seed, cfg)
    # load LoRA adapter weights
    ap = os.path.join(model_dir, "lora", "adapter_model.safetensors")
    if os.path.exists(ap):
        sd = load_file(ap)
        set_peft_model_state_dict(model.peft, sd)
    # load heads
    hp = os.path.join(model_dir, "heads.pt")
    if os.path.exists(hp):
        model.load_state_dict(torch.load(hp, map_location="cuda"), strict=False)
    model.eval()
    return model, proc, tok


@torch.no_grad()
def score_split(model, proc, images, attrs, config, seed):
    scores = []
    control = T.shuffled_attr_map(images, attrs, seed) if config == "shuffled" else attrs
    for im in images:
        img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
        px = proc.image_processor(img, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        a = torch.tensor(attrs.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        a_control = torch.tensor(control.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        s = float(model.classify(px, a, a_control=a_control)[0])
        scores.append(s)
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = os.path.basename(args.model.rstrip("/"))

    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    man = man[man["y"] >= 0]
    attrs = T.load_attrs()
    model, proc, tok = load_trained(args.model, args.config, args.seed)

    va = man[man.split == "val"]; te = man[man.split == "test"]
    sva = score_split(model, proc, va.image.tolist(), attrs, args.config, args.seed); yva = va.y.to_numpy()
    ste = score_split(model, proc, te.image.tolist(), attrs, args.config, args.seed); yte = te.y.to_numpy()

    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
    fpr, tpr, thr = roc_curve(yva, sva); j = tpr - fpr
    thr_star = float(thr[np.argmax(j)])
    result = pd.DataFrame({
        "image": te.image.tolist(), "y": yte, "score": ste,
        "pred": (ste >= thr_star).astype(int),
    })
    for column in ("patient_id", "lesion_id"):
        if column in te.columns:
            result[column] = te[column].to_numpy()
    result.to_csv(os.path.join(OUT, f"{name}_test.csv"), index=False)
    with open(os.path.join(OUT, f"{name}_thr.json"), "w") as f:
        json.dump({"threshold": thr_star, "config": args.config}, f)
    print(f"{name}: test AUROC={roc_auc_score(yte, ste):.3f} "
          f"AUPRC={average_precision_score(yte, ste):.3f} thr={thr_star:.3f} n={len(yte)}")


if __name__ == "__main__":
    main()
