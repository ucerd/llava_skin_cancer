"""
LLaVA prompt-based classification baselines:
  - Zero-shot LLaVA-1.5-7B (base, no fine-tuning)
  - Standard LLaVA-LoRA (the existing ../llava-finetuned adapter)
Malignancy score = P('yes') / (P('yes') + P('no')) for the prompt asking whether the
lesion is malignant, read from next-token logits. Threshold selected on validation.
Runs after VLM training completes (same GPU, no contention).
"""
import os, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
BASE = os.path.join(ROOT, "llava-base-model")
ADAPTER = os.path.join(ROOT, "llava-finetuned")

PROMPT = "USER: <image>\nIs this skin lesion malignant? Answer yes or no.\nASSISTANT:"


def load(processor_only=False, adapter=False):
    processor = AutoProcessor.from_pretrained(BASE)
    if processor_only:
        return processor, None
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    if adapter and os.path.exists(ADAPTER):
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return processor, model


@torch.no_grad()
def yes_prob(model, processor, img, yes_ids, no_ids):
    inp = processor(text=PROMPT, images=img, return_tensors="pt").to("cuda", torch.bfloat16)
    inp["input_ids"] = inp["input_ids"].long()
    logits = model(**inp).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    py = float(p[yes_ids].sum()); pn = float(p[no_ids].sum())
    return py / (py + pn + 1e-8)


def run(tag, adapter):
    processor, model = load(adapter=adapter)
    tok = processor.tokenizer
    yes_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in ["yes", "Yes", "▁yes", "▁Yes"]
               if tok(w, add_special_tokens=False).input_ids]
    no_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in ["no", "No", "▁no", "▁No"]
              if tok(w, add_special_tokens=False).input_ids]
    man = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
    man = man[man["y"] >= 0]
    def score(split):
        d = man[man.split == split]; out = []
        for im in d.image.tolist():
            img = Image.open(os.path.join(IMG_DIR, im)).convert("RGB")
            out.append(yes_prob(model, processor, img, yes_ids, no_ids))
        return np.array(out), d.y.to_numpy(), d.image.tolist()
    sva, yva, _ = score("val"); ste, yte, imte = score("test")
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
    result.to_csv(os.path.join(OUT, f"baseline_{tag}.csv"), index=False)
    json.dump({"threshold": thr_star}, open(os.path.join(OUT, f"baseline_{tag}_thr.json"), "w"))
    print(f"{tag}: AUROC={roc_auc_score(yte, ste):.3f} AUPRC={average_precision_score(yte, ste):.3f} "
          f"thr={thr_star:.3f} n={len(yte)}")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    run("llava_zeroshot", adapter=False)
    run("llava_lora", adapter=True)
