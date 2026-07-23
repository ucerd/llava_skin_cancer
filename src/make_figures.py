"""
Generate figures and LaTeX tables from the evaluation outputs. Every value traces
to a frozen prediction/output file.
"""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

PRETTY = {"full_s0": "DermaAttr-VLM (predicted attrs)", "oracle_s0": "DermaAttr-VLM (oracle attrs)",
          "no_attr_s0": "No-attribute ablation", "shuffled_s0": "Shuffled-attribute control",
          "no_gate_s0": "No-gate", "no_rec_s0": "No-reconstruction", "no_dist_s0": "No-distillation",
          "resnet50": "ResNet-50 (linear probe)", "convnext_tiny": "ConvNeXt-Tiny (linear probe)",
          "swin_tiny": "Swin-Tiny (linear probe)", "llava_lora": "Standard LLaVA-LoRA",
          "llava_zeroshot": "Zero-shot LLaVA"}


def score_files():
    d = {}
    invalid_path = os.path.join(OUT, "invalidated_outputs.json")
    invalid = set(json.load(open(invalid_path))) if os.path.exists(invalid_path) else set()
    for p in glob.glob(os.path.join(OUT, "*_test.csv")) + glob.glob(os.path.join(OUT, "baseline_*.csv")):
        if "_meta" in p or "_thr" in p:
            continue
        name = os.path.basename(p).replace("_test.csv", "").replace("baseline_", "").replace(".csv", "")
        if name in invalid:
            continue
        df = pd.read_csv(p)
        if {"y", "score"}.issubset(df.columns):
            d[name] = df
    return d


def fig_training_curves():
    p = os.path.join(HERE, "models", "full_s0", "step_log.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    w = 5  # stated smoothing window
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for col, lab in [("loss", "total"), ("lm", "language-model")]:
        if col in df:
            s = df[col].rolling(w, min_periods=1).mean()
            ax[0].plot(df.step, df[col], alpha=0.25, lw=0.8)
            ax[0].plot(df.step, s, label=f"{lab} (window={w})", lw=1.6)
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Training loss")
    for col in ["cls", "rec"]:
        if col in df:
            ax[1].plot(df.step, df[col].rolling(w, min_periods=1).mean(), label=col, lw=1.6)
    ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("auxiliary loss"); ax[1].legend()
    ax[1].set_title("Classification & reconstruction losses")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "training_curves.png"), dpi=150)
    plt.close()


def fig_roc(scores):
    plt.figure(figsize=(6.5, 6))
    order = sorted(scores.items(), key=lambda kv: -roc_auc_score(kv[1].y, kv[1].score))
    for name, df in order:
        y, s = df.y.to_numpy(), df.score.to_numpy()
        fpr, tpr, _ = roc_curve(y, s); a = roc_auc_score(y, s)
        plt.plot(fpr, tpr, lw=1.8, label=f"{PRETTY.get(name, name)} (AUROC={a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC: malignant vs benign")
    plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc.png"), dpi=150); plt.close()


def fig_calibration(scores):
    plt.figure(figsize=(6, 6))
    for name in ["full_s0", "oracle_s0", "resnet50"]:
        if name not in scores:
            continue
        df = scores[name]; y, s = df.y.to_numpy(), df.score.to_numpy()
        bins = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            m = (s >= bins[i]) & (s < bins[i + 1] if i < 9 else s <= bins[i + 1])
            if m.sum():
                xs.append(s[m].mean()); ys.append(y[m].mean())
        plt.plot(xs, ys, "o-", label=PRETTY.get(name, name))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
    plt.title("Calibration"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "calibration.png"), dpi=150); plt.close()


def fig_generation():
    files = glob.glob(os.path.join(OUT, "gen_*_bycat.csv"))
    if not files:
        return
    plt.figure(figsize=(9, 4.5))
    dfs = {os.path.basename(f).replace("gen_", "").replace("_bycat.csv", ""): pd.read_csv(f) for f in files}
    cats = sorted(set().union(*[set(d.category) for d in dfs.values()]))
    x = np.arange(len(cats)); w = 0.8 / max(len(dfs), 1)
    for i, (name, d) in enumerate(dfs.items()):
        vals = [float(d[d.category == c].bleu4.mean()) if (d.category == c).any() else 0 for c in cats]
        plt.bar(x + i * w, vals, w, label=PRETTY.get(name, name))
    plt.xticks(x + 0.4, cats, rotation=30, ha="right", fontsize=8)
    plt.ylabel("BLEU-4"); plt.title("Generation quality by question category")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "generation_by_category.png"), dpi=150); plt.close()


def tables(scores):
    # classification table from stats.py output if present, else compute
    sp = os.path.join(OUT, "classification_stats.csv")
    if os.path.exists(sp):
        df = pd.read_csv(sp)
        cols = ["model", "n", "auroc", "auroc_lo", "auroc_hi", "auprc", "sens", "spec", "f1", "brier", "ece"]
        df = df[cols].copy()
        df["model"] = df["model"].map(lambda m: PRETTY.get(m, m))
        with open(os.path.join(TAB, "classification.tex"), "w") as f:
            f.write(df.to_latex(index=False, float_format="%.3f",
                    caption="Binary malignant-vs-benign performance on the held-out test set "
                    "(image-level bootstrap 95\\% CIs). No patient_id is available for this subset, "
                    "so patient-level clustering could not be applied.",
                    label="tab:classification_real"))
    pp = os.path.join(OUT, "pairwise_stats.csv")
    if os.path.exists(pp):
        dfp = pd.read_csv(pp)
        dfp["other"] = dfp["other"].map(lambda m: PRETTY.get(m, m))
        with open(os.path.join(TAB, "pairwise.tex"), "w") as f:
            f.write(dfp.to_latex(index=False, float_format="%.3f",
                    caption="Pairwise DeLong (AUROC) and McNemar (paired correctness) tests vs "
                    "DermaAttr-VLM (predicted attrs).", label="tab:pairwise_real"))
    gens = glob.glob(os.path.join(OUT, "gen_*_overall.json"))
    if gens:
        rows = [json.load(open(g)) for g in gens]
        dg = pd.DataFrame(rows); dg["model"] = dg["model"].map(lambda m: PRETTY.get(m, m))
        with open(os.path.join(TAB, "generation.tex"), "w") as f:
            f.write(dg.to_latex(index=False, float_format="%.3f",
                    caption="Generation metrics on a fixed test subset.",
                    label="tab:generation_real"))


def main():
    scores = score_files()
    fig_training_curves()
    if scores:
        fig_roc(scores); fig_calibration(scores)
    fig_generation()
    tables(scores)
    print("figures ->", os.listdir(FIG))
    print("tables  ->", os.listdir(TAB))


if __name__ == "__main__":
    main()
