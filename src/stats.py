"""
Statistics harness for frozen case-level predictions.

Important invariants:
  - Point estimates are calculated on the original test cases. Bootstrap samples are
    used only for confidence intervals.
  - Threshold-dependent metrics and McNemar tests use each model's validation-selected
    threshold (or the stored ``pred`` column produced with that threshold).
  - When ``patient_id``, ``lesion_id``, or ``cluster_id`` is available, bootstrap
    resampling is performed at that cluster level. Otherwise the output is explicitly
    labelled image-level and must not be described as patient-level inference.
"""
import os, json, glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
rng = np.random.default_rng(12345)

# ---------- fast DeLong ----------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_var(scores, labels):
    order = np.argsort(-labels)  # positives first
    labels = labels[order]; scores = scores[order]
    m = int(labels.sum()); n = len(labels) - m
    pos = scores[:m]; neg = scores[m:]
    tx = _compute_midrank(pos); ty = _compute_midrank(neg); tz = _compute_midrank(scores)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    return auc, v01, v10, m, n

def delong_test(s1, s2, y):
    y = y.astype(int)
    a1, v01_1, v10_1, m, n = delong_var(s1, y)
    a2, v01_2, v10_2, _, _ = delong_var(s2, y)
    sx = np.cov(np.vstack([v01_1, v01_2])); sy = np.cov(np.vstack([v10_1, v10_2]))
    var = sx / m + sy / n
    z_var = var[0, 0] + var[1, 1] - 2 * var[0, 1]
    if z_var <= 0:
        return a1, a2, 1.0
    from scipy.stats import norm
    z = (a1 - a2) / np.sqrt(z_var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return a1, a2, float(p)

def mcnemar(correct1, correct2):
    """correct1/2: boolean arrays of per-case correctness. Returns (b, c, p)."""
    b = int(np.sum(correct1 & ~correct2))  # 1 right, 2 wrong
    c = int(np.sum(~correct1 & correct2))  # 1 wrong, 2 right
    from scipy.stats import binomtest
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue
    return b, c, float(p)

def boot_ci(fn, y, s, clusters=None, n_boot=2000):
    """Return the observed point estimate and a percentile bootstrap CI."""
    vals = []
    idx = np.arange(len(y), dtype=int)
    if clusters is None:
        cluster_values = idx
        members = {int(i): np.array([i], dtype=int) for i in idx}
    else:
        clusters = np.asarray(clusters)
        cluster_values = np.unique(clusters)
        members = {c: np.flatnonzero(clusters == c) for c in cluster_values}
    for _ in range(n_boot):
        sampled = rng.choice(cluster_values, len(cluster_values), replace=True)
        bi = np.concatenate([members[c] for c in sampled])
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(fn(y[bi], s[bi]))
    if not vals:
        raise RuntimeError("No valid bootstrap resamples contained both classes.")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(fn(y, s)), float(lo), float(hi)

def ece(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; pts = []
    for i in range(bins):
        m = (s >= edges[i]) & (s < edges[i + 1] if i < bins - 1 else s <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = s[m].mean(); acc = y[m].mean(); w = m.mean()
        e += w * abs(acc - conf); pts.append([float(conf), float(acc), int(m.sum())])
    return float(e), pts

def clf_metrics(y, s, thr=None, pred=None):
    if pred is None:
        if thr is None:
            raise ValueError("Either thr or pred must be supplied.")
        pred = (s >= thr).astype(int)
    else:
        pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    acc = (tp + tn) / len(y)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, sens=sens, spec=spec, prec=prec, f1=f1)

def load_scores():
    d = {}
    invalid_path = os.path.join(OUT, "invalidated_outputs.json")
    invalid = set(json.load(open(invalid_path))) if os.path.exists(invalid_path) else set()
    for p in sorted(glob.glob(os.path.join(OUT, "*_test.csv"))) + \
             sorted(glob.glob(os.path.join(OUT, "baseline_*.csv"))):
        if p.endswith("_meta.json"):
            continue
        name = os.path.basename(p).replace("_test.csv", "").replace("baseline_", "").replace(".csv", "")
        if name in invalid:
            continue
        df = pd.read_csv(p)
        if {"y", "score"}.issubset(df.columns):
            d[name] = df
    return d


def threshold_for(name, df):
    """Resolve the validation-selected threshold despite legacy filename variants."""
    candidates = [
        os.path.join(OUT, f"{name}_thr.json"),
        os.path.join(OUT, f"baseline_{name}_thr.json"),
        os.path.join(OUT, f"{name}_meta.json"),
        os.path.join(OUT, f"baseline_{name}_meta.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                value = json.load(f).get("threshold")
            if value is not None:
                return float(value)
    if "pred" in df.columns:
        # The classifications remain valid, but a unique numeric threshold cannot always
        # be recovered from rounded scores. Refuse to invent one for the table.
        return np.nan
    raise FileNotFoundError(f"No validation threshold found for {name}.")


def cluster_column(df):
    for col in ("patient_id", "lesion_id", "cluster_id"):
        if col in df.columns and df[col].notna().all():
            return col
    return None

def main(ref="full_s0", std_baseline="resnet50"):
    scores = load_scores()
    if not scores:
        print("No score files yet in outputs/."); return
    rows = []
    for name, df in scores.items():
        y = df.y.to_numpy().astype(int); s = df.score.to_numpy(float)
        thr = threshold_for(name, df)
        ccol = cluster_column(df)
        clusters = df[ccol].to_numpy() if ccol else None
        auroc, alo, ahi = boot_ci(roc_auc_score, y, s, clusters=clusters)
        auprc, plo, phi = boot_ci(average_precision_score, y, s, clusters=clusters)
        stored_pred = df.pred.to_numpy(int) if "pred" in df.columns else None
        cm = clf_metrics(y, s, thr=thr, pred=stored_pred)
        brier = brier_score_loss(y, s); e, _ = ece(y, s)
        rows.append(dict(model=name, n=len(y), prevalence=float(y.mean()),
                         ci_unit=ccol or "image",
                         auroc=auroc, auroc_lo=alo, auroc_hi=ahi,
                         auprc=auprc, auprc_lo=plo, auprc_hi=phi,
                         brier=brier, ece=e, threshold=thr, **cm))
    res = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    res.to_csv(os.path.join(OUT, "classification_stats.csv"), index=False)
    seeded = res[res.model.str.contains(r"_s\d+$", regex=True)].copy()
    if not seeded.empty:
        seeded["configuration"] = seeded.model.str.replace(r"_s\d+$", "", regex=True)
        summary = seeded.groupby("configuration").agg(
            seeds=("model", "count"),
            auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
            brier_mean=("brier", "mean"), brier_sd=("brier", "std"),
        ).reset_index()
        summary.to_csv(os.path.join(OUT, "multiseed_stats.csv"), index=False)
    print(res[["model", "n", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier", "ece",
               "sens", "spec", "f1"]].to_string(index=False))

    # pairwise DeLong + McNemar vs reference, aligned on common images
    if ref in scores:
        rdf = scores[ref].set_index("image")
        comp = []
        for name, df in scores.items():
            if name == ref:
                continue
            m = df.set_index("image").join(rdf, lsuffix="_o", rsuffix="_r", how="inner").dropna()
            if len(m) < 10:
                continue
            if not np.array_equal(m["y_o"].to_numpy(), m["y_r"].to_numpy()):
                raise ValueError(f"Label mismatch after case alignment: {ref} vs {name}")
            y = m["y_o"].to_numpy().astype(int)
            a_ref, a_oth, p = delong_test(m["score_r"].to_numpy(float), m["score_o"].to_numpy(float), y)
            if "pred_r" in m.columns:
                pred_r = m["pred_r"].to_numpy(int)
            else:
                pred_r = (m["score_r"].to_numpy(float) >= threshold_for(ref, scores[ref])).astype(int)
            if "pred_o" in m.columns:
                pred_o = m["pred_o"].to_numpy(int)
            else:
                pred_o = (m["score_o"].to_numpy(float) >= threshold_for(name, scores[name])).astype(int)
            corr_r = pred_r == y
            corr_o = pred_o == y
            b, c, pm = mcnemar(corr_r, corr_o)
            comp.append(dict(reference=ref, other=name, n=len(m),
                             auroc_ref=a_ref, auroc_other=a_oth, delong_p=p,
                             mcnemar_b=b, mcnemar_c=c, mcnemar_p=pm))
        comp_df = pd.DataFrame(comp)
        if not comp_df.empty:
            comp_df["delong_p_holm"] = np.nan
            comp_df["mcnemar_p_holm"] = np.nan
            confirmatory = comp_df["other"] != "llava_lora"
            comp_df.loc[confirmatory, "delong_p_holm"] = multipletests(
                comp_df.loc[confirmatory, "delong_p"].to_numpy(float), method="holm"
            )[1]
            comp_df.loc[confirmatory, "mcnemar_p_holm"] = multipletests(
                comp_df.loc[confirmatory, "mcnemar_p"].to_numpy(float), method="holm"
            )[1]
        comp_df.to_csv(os.path.join(OUT, "pairwise_stats.csv"), index=False)
        print("\n=== pairwise vs", ref, "===")
        if comp:
            print(comp_df[["other", "auroc_ref", "auroc_other", "delong_p",
                           "delong_p_holm", "mcnemar_b", "mcnemar_c",
                           "mcnemar_p", "mcnemar_p_holm"]].to_string(index=False))

if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3] if len(sys.argv) > 1 else []))
