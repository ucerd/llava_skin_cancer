"""
Image-only classification baselines for the binary malignant-vs-benign endpoint. Uses ImageNet-pretrained backbones as frozen feature extractors
with a class-weighted logistic-regression head (linear probe). Runs on CPU so it does
not contend with the GPU VLM training.

Backbones: ResNet-50, ConvNeXt-Tiny, Swin-Tiny (modern classifiers).
Threshold selected on validation split; case-level test scores saved for the shared
statistics harness. Same frozen split and preprocessing for every model.
"""
import os, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def backbone(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = torch.nn.Identity()
    elif name == "swin_tiny":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1); m.head = torch.nn.Identity()
    else:
        raise ValueError(name)
    return m.eval()

@torch.no_grad()
def extract(name, images):
    m = backbone(name)
    feats = []
    for i, im in enumerate(images):
        img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
        f = m(tfm(img).unsqueeze(0)).squeeze(0).numpy()
        feats.append(f)
        if (i + 1) % 100 == 0:
            print(f"  {name}: {i+1}/{len(images)}")
    return np.array(feats, np.float32)

def main():
    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    man = man[man["y"] >= 0].copy()  # exclude indeterminate from classification
    tr = man[man.split == "train"]; va = man[man.split == "val"]; te = man[man.split == "test"]
    for name in ["resnet50", "convnext_tiny", "swin_tiny"]:
        print(f"=== {name} ===")
        Xtr = extract(name, tr.image.tolist()); ytr = tr.y.to_numpy()
        Xva = extract(name, va.image.tolist()); yva = va.y.to_numpy()
        Xte = extract(name, te.image.tolist()); yte = te.y.to_numpy()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr)
        sva = clf.predict_proba(Xva)[:, 1]; ste = clf.predict_proba(Xte)[:, 1]
        # threshold: maximize Youden J on validation
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(yva, sva)
        j = tpr - fpr; thr_star = float(thr[np.argmax(j)])
        result = pd.DataFrame({
            "image": te.image.tolist(), "y": yte, "score": ste,
            "pred": (ste >= thr_star).astype(int),
        })
        for column in ("patient_id", "lesion_id"):
            if column in te.columns:
                result[column] = te[column].to_numpy()
        result.to_csv(os.path.join(OUT, f"baseline_{name}.csv"), index=False)
        from sklearn.metrics import roc_auc_score, average_precision_score
        print(f"  {name}: test AUROC={roc_auc_score(yte, ste):.3f} "
              f"AUPRC={average_precision_score(yte, ste):.3f} thr={thr_star:.3f}")
        with open(os.path.join(OUT, f"baseline_{name}_meta.json"), "w") as f:
            json.dump({"model": name, "type": "frozen-features + class-weighted logistic (linear probe)",
                       "threshold": thr_star, "n_test": int(len(yte))}, f, indent=2)
    print("baselines done")

if __name__ == "__main__":
    main()
