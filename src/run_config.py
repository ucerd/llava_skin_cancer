"""
Single-process runner: load the model once, then train + classify-eval + generate-eval
for one config, loading the model once per job.

Usage: python run_config.py --config full --seed 0 --epochs 1 --per_image_cap 3 --gen_n 150
"""
import os, sys, json, csv, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_derma as TD
from derma_attr_vlm import D_ATTR
import pandas as pd

IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def train(model, tok, proc, cfg_name, args, out):
    prefix_ids = torch.tensor(tok("USER: ", add_special_tokens=True).input_ids)
    pad = tok.pad_token_id or tok.eos_token_id
    coll = TD.make_collate(prefix_ids, pad)
    attrs = TD.load_attrs()
    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    train_images = man[man.split == "train"].image.tolist()
    fusion_attrs = TD.shuffled_attr_map(train_images, attrs, args.seed) if cfg_name == "shuffled" else attrs
    tr = TD.QADataset("train", proc, attrs, TD.load_labels(), tok,
                      per_image_cap=args.per_image_cap, fusion_attrs=fusion_attrs)
    dl = DataLoader(tr, batch_size=1, shuffle=True, collate_fn=coll, num_workers=0)
    params = [p for p in model.peft.parameters() if p.requires_grad]
    for m in model.trainable_modules():
        params += list(m.parameters())
    params += [model.Ws, model.delta]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=1e-2)
    total = (len(dl) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1), eta_min=1e-6)
    logf = open(os.path.join(out, "step_log.csv"), "w", newline=""); lw = csv.writer(logf)
    lw.writerow(["step", "epoch", "loss", "lm", "cls", "rec", "dist", "ahead", "lr"])
    gstep = 0; t0 = time.time(); model.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        for it, batch in enumerate(dl):
            px, a, a_control, y, mid, ans, ml, al = batch
            px = px.cuda().to(torch.bfloat16); a = a.cuda(); a_control = a_control.cuda(); y = y.cuda()
            loss, _, logs = model(
                px, a, y, prefix_ids, mid, ans, ml, al,
                a_control=a_control
            )
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                lw.writerow([gstep, ep, float(loss), logs.get("lm"), logs.get("cls"),
                             logs.get("rec"), logs.get("dist"), logs.get("ahead"),
                             sched.get_last_lr()[0]]); logf.flush()
                if gstep % 20 == 0:
                    print(f"[{cfg_name}] ep{ep} step{gstep} loss={float(loss):.3f} "
                          f"{time.time()-t0:.0f}s", flush=True)
        # per-epoch checkpoint so a wall-time timeout still leaves a usable model
        _save(model, out, cfg_name, args, gstep, len(tr), t0, ep + 1)
        print(f"[{cfg_name}] checkpoint saved after epoch {ep} (step {gstep})", flush=True)
    logf.close()
    _save(model, out, cfg_name, args, gstep, len(tr), t0, args.epochs)
    print(f"[{cfg_name}] trained {gstep} steps in {time.time()-t0:.0f}s", flush=True)


def _save(model, out, cfg_name, args, gstep, n_items, t0, epochs_done):
    model.peft.save_pretrained(os.path.join(out, "lora"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("peft.")},
               os.path.join(out, "heads.pt"))
    json.dump({"config": cfg_name, "epochs_done": epochs_done, "steps": gstep,
               "train_items": n_items, "wall_seconds": time.time() - t0},
              open(os.path.join(out, "config.json"), "w"), indent=2)


@torch.no_grad()
def eval_classify(model, proc, base, cfg_name, seed):
    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv")); man = man[man.y >= 0]
    attrs = TD.load_attrs()
    def sc(split):
        d = man[man.split == split]; out = []
        control = TD.shuffled_attr_map(d.image.tolist(), attrs, seed) if cfg_name == "shuffled" else attrs
        for im in d.image.tolist():
            img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
            px = proc.image_processor(img, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
            av = torch.tensor(attrs.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
            ac = torch.tensor(control.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
            out.append(float(model.classify(px, av, a_control=ac)[0]))
        return np.array(out), d.y.to_numpy(), d.image.tolist()
    sva, yva, _ = sc("val"); ste, yte, imte = sc("test")
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
    fpr, tpr, thr = roc_curve(yva, sva); thr_star = float(thr[np.argmax(tpr - fpr)])
    test_rows = man[man.split == "test"]
    result = pd.DataFrame({
        "image": imte, "y": yte, "score": ste,
        "pred": (ste >= thr_star).astype(int),
    })
    for column in ("patient_id", "lesion_id"):
        if column in test_rows.columns:
            result[column] = test_rows[column].to_numpy()
    result.to_csv(os.path.join(OUT, f"{base}_test.csv"), index=False)
    json.dump({"threshold": thr_star}, open(os.path.join(OUT, f"{base}_thr.json"), "w"))
    print(f"[{base}] CLS AUROC={roc_auc_score(yte, ste):.3f} AUPRC={average_precision_score(yte, ste):.3f}", flush=True)


@torch.no_grad()
def eval_generate(model, proc, tok, base, n, cfg_name, seed):
    from eval_generate import categorize, rouge_l, smooth, stratified_items
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    items = [json.loads(l) for l in open(os.path.join(DATA, "qa_test.jsonl")) if l.strip()]
    items = [d for d in items if TD._qa_from_conv(d)]
    items = stratified_items(items, n, seed=42)
    attrs = TD.load_attrs()
    images = [os.path.basename(d["image"]) for d in items]
    control = TD.shuffled_attr_map(images, attrs, seed) if cfg_name == "shuffled" else attrs
    prefix_ids = torch.tensor(tok("USER: ", add_special_tokens=True).input_ids)
    rows = []
    for d in items:
        im = os.path.basename(d["image"])
        img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
        px = proc.image_processor(img, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        av = torch.tensor(attrs.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        ac = torch.tensor(control.get(im, np.zeros(D_ATTR, np.float32))).unsqueeze(0).cuda()
        qa = TD._qa_from_conv(d)
        if qa is None:
            continue
        q = qa[0].replace("<image>", "").strip(); ref = qa[1].strip()
        mid = tok(q + "\nASSISTANT: ", add_special_tokens=False).input_ids[:160]
        pred = model.generate(
            px, av, prefix_ids, torch.tensor(mid).unsqueeze(0), len(mid), 80,
            a_control=ac
        ).strip()
        rt, ht = ref.lower().split(), pred.lower().split()
        b4 = sentence_bleu([rt], ht, weights=(.25, .25, .25, .25), smoothing_function=smooth) if ht else 0.0
        try: met = meteor_score([rt], ht)
        except Exception: met = 0.0
        rows.append(dict(image=im, category=categorize(q), reference=ref, prediction=pred,
                         bleu4=b4, rougeL=rouge_l(ref.lower(), pred.lower()), meteor=met))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, f"gen_{base}.csv"), index=False)
    df.groupby("category")[["bleu4", "rougeL", "meteor"]].mean().reset_index().to_csv(
        os.path.join(OUT, f"gen_{base}_bycat.csv"), index=False)
    ov = dict(model=base, n=len(df), bleu4=float(df.bleu4.mean()),
              rougeL=float(df.rougeL.mean()), meteor=float(df.meteor.mean()))
    json.dump(ov, open(os.path.join(OUT, f"gen_{base}_overall.json"), "w"), indent=2)
    print(f"[{base}] GEN {ov}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_image_cap", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--gen_n", type=int, default=150)
    ap.add_argument("--gen", action="store_true")
    args = ap.parse_args()
    cfg = TD.CONFIGS[args.config]
    base = f"{args.config}_s{args.seed}"
    out = os.path.join(HERE, "models", base); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    model, proc, tok = TD.build_model(args.seed, cfg)
    TD.set_balanced_class_weights(model)
    print(f"[{base}] model loaded in {time.time()-t0:.0f}s", flush=True)
    train(model, tok, proc, args.config, args, out)
    model.eval()
    eval_classify(model, proc, base, args.config, args.seed)
    if args.gen:
        eval_generate(model, proc, tok, base, args.gen_n, args.config, args.seed)
    print(f"[{base}] ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
