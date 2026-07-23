"""
Generate the error-analysis and qualitative-example figures from measured outputs.
  - error_analysis.png : confusion matrix of the deployable model + false-positive /
                         false-negative composition across models.
  - qualitative_examples.png : real dermoscopic images with the model's generated answer
                               and the reference answer.
Figures are written to figures/ and copied into the manuscript.
"""
import os, textwrap, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
MANU_FIG = os.path.join(os.path.dirname(ROOT), "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)

PRETTY = {"full_s0": "DermaAttr-VLM", "resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-Tiny",
          "swin_tiny": "Swin-Tiny", "llava_lora": "LLaVA-LoRA", "llava_zeroshot": "Zero-shot LLaVA"}
BLUE, ORANGE = "#3b6fb0", "#d1893f"   # colourblind-safe pair


def fig_error_analysis():
    d = pd.read_csv(os.path.join(OUT, "classification_stats.csv")).set_index("model")
    order = ["full_s0", "resnet50", "convnext_tiny", "swin_tiny", "llava_lora", "llava_zeroshot"]
    order = [m for m in order if m in d.index]

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.35], wspace=0.35)

    # (a) confusion matrix of the deployable model
    ax0 = fig.add_subplot(gs[0])
    r = d.loc["full_s0"]
    cm = np.array([[int(r.tn), int(r.fp)], [int(r.fn), int(r.tp)]])
    im = ax0.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["Benign", "Malignant"])
    ax0.set_yticks([0, 1]); ax0.set_yticklabels(["Benign", "Malignant"])
    ax0.set_xlabel("Predicted"); ax0.set_ylabel("Actual")
    ax0.set_title("(a) DermaAttr-VLM confusion matrix")

    # (b) false-positive / false-negative composition across models
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(order)); w = 0.38
    fp = [int(d.loc[m].fp) for m in order]
    fn = [int(d.loc[m].fn) for m in order]
    ax1.bar(x - w / 2, fp, w, label="False positives", color=BLUE)
    ax1.bar(x + w / 2, fn, w, label="False negatives", color=ORANGE)
    for xi, (a, b) in enumerate(zip(fp, fn)):
        ax1.text(xi - w / 2, a + 0.4, str(a), ha="center", fontsize=8)
        ax1.text(xi + w / 2, b + 0.4, str(b), ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels([PRETTY[m] for m in order], rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Test-set errors (count)")
    ax1.set_title("(b) Error composition by model")
    ax1.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    plt.tight_layout()
    p = os.path.join(FIG, "error_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def _pick_examples(df, n=3):
    """Pick diverse, readable examples across categories."""
    df = df[df.prediction.str.len().between(60, 320)].copy()
    picks, used = [], set()
    for cat in ["diagnosis", "attribute_description", "feature_recognition", "pattern_description"]:
        sub = df[(df.category == cat) & (~df.image.isin(used))]
        sub = sub[sub.reference.str.len() > 40]
        if len(sub):
            row = sub.sort_values("bleu4", ascending=False).iloc[0]
            picks.append(row); used.add(row.image)
        if len(picks) == n:
            break
    return picks


def _crop_43(im):
    """Center-crop to 4:3 so every image displays at identical dimensions."""
    w, h = im.size; ar = 4 / 3
    if w / h > ar:
        nw = int(h * ar); return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / ar); return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def fig_qualitative():
    df = pd.read_csv(os.path.join(OUT, "gen_full_s0.csv"))
    picks = _pick_examples(df, 3)
    n = len(picks)
    WRAP, FS = 112, 9
    fig = plt.figure(figsize=(11, 2.75 * n))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    gs = gridspec.GridSpec(n, 2, width_ratios=[0.72, 3.25], wspace=0.03, hspace=0.14)

    def wfrac(ax, s, weight):
        tt = ax.text(0, -9, s, fontsize=FS, fontweight=weight, transform=ax.transAxes)
        bb = tt.get_window_extent(renderer=rend); tt.remove()
        inv = ax.transAxes.inverted()
        return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

    def put(ax, lines, y, dy, color, weight, justify):
        """Draw wrapped lines top-aligned; fully justify every line except the last."""
        sp = wfrac(ax, " ", weight)
        for k, ln in enumerate(lines):
            words = ln.split()
            if justify and k < len(lines) - 1 and len(words) > 1:
                widths = [wfrac(ax, w, weight) for w in words]
                gap = (1.0 - sum(widths)) / (len(words) - 1)
                gap = min(max(gap, sp), 4 * sp)
                x = 0.0
                for w, wd in zip(words, widths):
                    ax.text(x, y, w, fontsize=FS, fontweight=weight, color=color,
                            va="top", transform=ax.transAxes)
                    x += wd + gap
            else:
                ax.text(0, y, ln, fontsize=FS, fontweight=weight, color=color,
                        va="top", transform=ax.transAxes)
            y -= dy
        return y

    for i, r in enumerate(picks):
        axi = fig.add_subplot(gs[i, 0])
        axi.imshow(_crop_43(Image.open(os.path.join(IMG_DIR, r.image)).convert("RGB")))
        axi.set_anchor("N"); axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(False)

        axt = fig.add_subplot(gs[i, 1]); axt.axis("off"); axt.set_anchor("N")
        dy = 0.072; y = 1.0
        axt.text(0, y, f"Question category: {r.category.replace('_', ' ').title()}",
                 fontsize=FS + 1, fontweight="bold", va="top", transform=axt.transAxes); y -= 1.5 * dy
        axt.text(0, y, "DermaAttr-VLM", fontsize=FS, fontweight="bold", color=BLUE,
                 va="top", transform=axt.transAxes); y -= dy
        y = put(axt, textwrap.wrap(str(r.prediction), WRAP), y, dy, "black", "normal", True)
        y -= 0.5 * dy
        axt.text(0, y, "Reference", fontsize=FS, fontweight="bold", color="#555",
                 va="top", transform=axt.transAxes); y -= dy
        y = put(axt, textwrap.wrap(str(r.reference), WRAP), y, dy, "#444", "normal", True)

    p = os.path.join(FIG, "qualitative_examples.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def main():
    for p in (fig_error_analysis(), fig_qualitative()):
        dst = os.path.join(MANU_FIG, os.path.basename(p))
        shutil.copy(p, dst)
        print("wrote", os.path.basename(p), "->", dst)


if __name__ == "__main__":
    main()
