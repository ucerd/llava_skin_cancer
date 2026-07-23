# Reproducibility Guide

## Reproducibility levels

### Level A: Verify the released data package

```bash
cd src/data
sha256sum -c checksums.sha256
```

Expected result: all four checked files report `OK`.

### Level B: Recreate statistical tables and figures

This level uses the frozen case-level outputs and does not require source images or model weights.

```bash
python src/stats.py full_s0 resnet50
python src/make_figures.py
python src/make_extra_figures.py
```

### Level C: Re-run evaluation

This level requires:

- the 500 source images under `images/`;
- the base model under `llava-base-model/`;
- trained LoRA and custom-head weights under the relevant `src/models/<run>/` directory.

```bash
python src/eval_classify.py --model src/models/full_s0 --config full --seed 0
python src/eval_generate.py --model src/models/full_s0 --config full --seed 0 --n 250
```

### Level D: Re-train a configuration

```bash
python src/train_derma.py \
  --config full \
  --epochs 3 \
  --seed 0 \
  --per_image_cap 6 \
  --accum 8 \
  --out src/models/full_s0
```

## Frozen release facts

- Split seed: 42.
- Training configuration seed: 0.
- Training images: 300.
- Validation images: 74.
- Test images: 126.
- Binary test images: 118.
- Training Q&A pairs before per-image cap: 5,999.
- Training pairs after the cap and malformed-entry filtering: 1,798.
- Epochs: 3.
- Updates per epoch: 224.
- Total updates: 672.

## Non-reproducible element

The original source-to-local filename mapping for the selected 500 images is not included. A full reconstruction from the upstream dataset therefore requires the original mapping used during cohort selection. The remaining metadata cannot uniquely identify the selected source record for each local image.

## Validation performed on this release

- Python source files compile successfully with `python -m compileall`.
- All JSONL files parse without malformed JSON records.
- The released split checksums validate.
- The split counts agree with `src/data/split_summary.json`.
