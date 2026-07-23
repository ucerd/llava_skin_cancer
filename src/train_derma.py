"""
Train DermaAttr-VLM (or an ablation config) on the frozen split.
Logs per-step losses to CSV for the training-curve figure.

Usage:
  python train_derma.py --config full --epochs 5 --out models/full_s0 --seed 0
  python train_derma.py --config full --smoke     # quick pipeline validation
Configs:
  full       : use_attr, attr_source=predicted, gate, rec, dist   (deployable primary)
  oracle     : attr_source=oracle (reference attrs at inference; upper bound)
  shuffled   : attr_source=shuffled (shortcut-dependence control)
  no_attr    : no attribute pathway (language+cls only)
  no_gate    : fusion without gate
  no_rec     : no reconstruction loss
  no_dist    : no distillation loss
"""
import os, sys, json, csv, time, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from derma_attr_vlm import DermaAttrVLM, ATTRS, D_ATTR

MODEL_PATH = os.path.join(ROOT, "llava-base-model")
IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
ATTR_CSV = os.path.join(HERE, "attributes", "attributes.csv")

CONFIGS = {
    "full":     dict(use_attr=True, attr_source="predicted", use_gate=True, use_rec=True, use_dist=True),
    "oracle":   dict(use_attr=True, attr_source="oracle",    use_gate=True, use_rec=True, use_dist=True),
    "shuffled": dict(use_attr=True, attr_source="shuffled",  use_gate=True, use_rec=True, use_dist=True),
    "no_attr":  dict(use_attr=False, attr_source="none",     use_gate=True, use_rec=True, use_dist=True),
    "no_gate":  dict(use_attr=True, attr_source="predicted", use_gate=False, use_rec=True, use_dist=True),
    "no_rec":   dict(use_attr=True, attr_source="predicted", use_gate=True, use_rec=False, use_dist=True),
    "no_dist":  dict(use_attr=True, attr_source="predicted", use_gate=True, use_rec=True, use_dist=False),
}
for c in CONFIGS.values():
    c.update(lam_cls=0.1, lam_rec=0.05, lam_dist=0.02)


def _qa_from_conv(d):
    """Robustly extract (question, answer) from a conversation; return None if malformed."""
    conv = d.get("conversations")
    if not isinstance(conv, list):
        return None
    q = a = None
    for turn in conv:
        if not isinstance(turn, dict) or "value" not in turn:
            continue
        who = turn.get("from")
        if who == "human" and q is None:
            q = turn["value"]
        elif who == "gpt" and a is None:
            a = turn["value"]
    if q is None or a is None or not str(a).strip():
        return None
    return str(q), str(a)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_attrs():
    import pandas as pd
    df = pd.read_csv(ATTR_CSV).set_index("image")
    return {im: df.loc[im, ATTRS].to_numpy(np.float32) for im in df.index}


def load_labels():
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    return {r["image"]: int(r["y"]) for _, r in df.iterrows()}


class QADataset(Dataset):
    def __init__(self, split, processor, attrs, labels, tok, limit=None,
                 per_image_cap=None, fusion_attrs=None):
        raw = []
        with open(os.path.join(DATA, f"qa_{split}.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
        if per_image_cap:  # keep all images, cap Q&A per image (bounded, stratified by image)
            per = {}
            self.items = []
            for d in raw:
                im = os.path.basename(d["image"])
                if per.get(im, 0) < per_image_cap:
                    self.items.append(d); per[im] = per.get(im, 0) + 1
        else:
            self.items = raw
        # drop malformed conversations (missing turns / 'value' keys) so training never crashes
        self.items = [d for d in self.items if _qa_from_conv(d) is not None
                      and os.path.exists(os.path.join(IMG_DIR, os.path.basename(d["image"])))]
        if limit:
            self.items = self.items[:limit]
        self.proc = processor
        self.attrs = attrs
        self.fusion_attrs = fusion_attrs or attrs
        self.labels = labels
        self.tok = tok

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        im = os.path.basename(d["image"])
        img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
        px = self.proc.image_processor(img, return_tensors="pt")["pixel_values"][0]
        q, ans = _qa_from_conv(d)
        q = q.replace("<image>", "").strip()
        mid = self.tok(q + "\nASSISTANT: ", add_special_tokens=False).input_ids
        answer = self.tok(ans, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        return dict(px=px, a=torch.tensor(self.attrs.get(im, np.zeros(D_ATTR, np.float32))),
                    a_control=torch.tensor(self.fusion_attrs.get(im, np.zeros(D_ATTR, np.float32))),
                    y=self.labels.get(im, -1), mid=mid, ans=answer)


def make_collate(prefix_ids, pad_id):
    def collate(batch):
        px = torch.stack([b["px"] for b in batch])
        a = torch.stack([b["a"] for b in batch])
        a_control = torch.stack([b["a_control"] for b in batch])
        y = torch.tensor([b["y"] for b in batch], dtype=torch.long)
        mids = [b["mid"] for b in batch]; anss = [b["ans"] for b in batch]
        # cap lengths for memory
        mids = [m[:160] for m in mids]; anss = [an[:120] for an in anss]
        Lm = max(len(m) for m in mids); La = max(len(an) for an in anss)
        mid_t = torch.full((len(batch), Lm), pad_id, dtype=torch.long)
        ans_t = torch.full((len(batch), La), pad_id, dtype=torch.long)
        mlen = torch.tensor([len(m) for m in mids]); alen = torch.tensor([len(an) for an in anss])
        for i, (m, an) in enumerate(zip(mids, anss)):
            mid_t[i, :len(m)] = torch.tensor(m); ans_t[i, :len(an)] = torch.tensor(an)
        return px, a, a_control, y, mid_t, ans_t, mlen, alen
    return collate


def shuffled_attr_map(images, attrs, seed=0):
    """Assign each image the attributes of a different image."""
    images = sorted(set(images))
    if len(images) < 2:
        raise ValueError("At least two images are required for a shuffled control.")
    offset = 1 + seed % (len(images) - 1)
    shifted = images[offset:] + images[:offset]
    return {image: attrs[other].copy() for image, other in zip(images, shifted)}


def set_balanced_class_weights(model):
    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    y = man[(man.split == "train") & (man.y >= 0)].y.to_numpy(int)
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2.0 * counts)
    model.class_weights.copy_(torch.tensor(weights, device=model.class_weights.device))


def build_model(seed, cfg):
    set_seed(seed)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tok = processor.tokenizer
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    llava = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    llava = prepare_model_for_kbit_training(
        llava, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    peft_model = get_peft_model(llava, lora)
    peft_model.enable_input_require_grads()
    model = DermaAttrVLM(peft_model, tok, cfg).cuda()
    # move custom modules to fp32 on gpu
    for mod in model.trainable_modules():
        mod.float().cuda()
    model.Ws.data = model.Ws.data.float().cuda(); model.delta.data = model.delta.data.float().cuda()
    return model, processor, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per_image_cap", type=int, default=None)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    out = args.out or os.path.join(HERE, "models", f"{args.config}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    model, processor, tok = build_model(args.seed, cfg)
    set_balanced_class_weights(model)

    prefix_ids = torch.tensor(tok("USER: ", add_special_tokens=True).input_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collate = make_collate(prefix_ids, pad_id)

    limit = 24 if args.smoke else args.limit
    attrs = load_attrs()
    train_manifest = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    train_images = train_manifest[train_manifest.split == "train"].image.tolist()
    fusion_attrs = shuffled_attr_map(train_images, attrs, args.seed) if args.config == "shuffled" else attrs
    tr = QADataset("train", processor, attrs, load_labels(), tok,
                   limit=limit, per_image_cap=args.per_image_cap, fusion_attrs=fusion_attrs)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=2)

    params = [p for p in model.peft.parameters() if p.requires_grad]
    for mod in model.trainable_modules():
        params += list(mod.parameters())
    params += [model.Ws, model.delta]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    total_steps = (len(dl) // args.accum) * (1 if args.smoke else args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=1e-6)

    log_path = os.path.join(out, "step_log.csv")
    logf = open(log_path, "w", newline=""); lw = csv.writer(logf)
    lw.writerow(["step", "epoch", "loss", "lm", "cls", "rec", "dist", "ahead", "lr"])

    epochs = 1 if args.smoke else args.epochs
    gstep = 0; t0 = time.time()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        for it, batch in enumerate(dl):
            px, a, a_control, y, mid_t, ans_t, mlen, alen = batch
            px = px.cuda().to(torch.bfloat16); a = a.cuda(); a_control = a_control.cuda(); y = y.cuda()
            loss, cls_logits, logs = model(
                px, a, y, prefix_ids, mid_t, ans_t, mlen, alen,
                a_control=a_control
            )
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                if gstep == 0:  # one-time sanity: confirm LoRA receives gradients
                    lg = [p.grad for n, p in model.peft.named_parameters()
                          if p.grad is not None and "lora" in n.lower()]
                    lnorm = sum(float(g.float().norm()) ** 2 for g in lg) ** 0.5 if lg else 0.0
                    print(f"[gradcheck] LoRA params with grad={len(lg)} gradnorm={lnorm:.3e} "
                          f"gamma_grad={None if model.fusion.gamma.grad is None else float(model.fusion.gamma.grad):.2e}",
                          flush=True)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                lw.writerow([gstep, ep, float(loss), logs.get("lm"), logs.get("cls"),
                             logs.get("rec"), logs.get("dist"), logs.get("ahead"),
                             sched.get_last_lr()[0]])
                logf.flush()
                if gstep % 10 == 0 or args.smoke:
                    el = time.time() - t0
                    print(f"[{args.config}] ep{ep} step{gstep} loss={float(loss):.3f} "
                          f"lm={logs.get('lm'):.3f} cls={logs.get('cls')} "
                          f"rec={logs.get('rec')} {el:.0f}s", flush=True)
        # checkpoint after each epoch (long runs are resumable/evaluable)
        model.peft.save_pretrained(os.path.join(out, "lora"))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()
                    if not k.startswith("peft.")}, os.path.join(out, "heads.pt"))
        print(f"[{args.config}] saved checkpoint after epoch {ep}", flush=True)
    logf.close()

    # save trainable state
    model.peft.save_pretrained(os.path.join(out, "lora"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()
                if not k.startswith("peft.")}, os.path.join(out, "heads.pt"))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"config": args.config, **cfg, "epochs": epochs, "seed": args.seed,
                   "train_items": len(tr), "steps": gstep,
                   "wall_seconds": time.time() - t0}, f, indent=2)
    print(f"DONE {args.config} seed{args.seed}: {gstep} steps, {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
