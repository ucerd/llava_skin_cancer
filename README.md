# DermaAttr-VLM

[![Code License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Derived%20Data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](environment.yml)
[![Paper status](https://img.shields.io/badge/Paper-Submitted-orange.svg)](#citation)

**DermaAttr-VLM** is an attribute-guided vision-language framework for dermoscopic image analysis. It combines CLIP ViT image features, deterministic dermoscopic attribute proxies, gated cross-modal fusion, and parameter-efficient adaptation of the LLaVA-1.5-7B/Vicuna language backbone.

This repository accompanies the manuscript:

> **DermaAttr-VLM: An Attribute-Guided Vision–Language Framework for Dermoscopic Image Analysis**

The repository is a research artifact. It is **not a medical device**, is **not intended for clinical diagnosis**, and must not be used to make treatment or management decisions.

## What is included

| Manuscript reproducibility item | Repository location |
|---|---|
| Frozen image-level train/validation/test split | `src/data/split_manifest.csv` |
| Split summary and SHA-256 checksums | `src/data/split_summary.json`, `src/data/checksums.sha256` |
| Split question-answer corpus | `src/data/qa_train.jsonl`, `src/data/qa_val.jsonl`, `src/data/qa_test.jsonl` |
| Unsplit question-answer source | `data.jsonl` |
| Algorithmic dermoscopic attribute values | `src/attributes/attributes.csv` |
| Attribute provenance and summary statistics | `src/attributes/attribute_report.json` |
| Dataset audit and split-generation code | `src/audit_dataset.py`, `src/freeze_splits.py` |
| Attribute extraction code | `src/extract_attributes.py` |
| DermaAttr-VLM architecture and training | `src/derma_attr_vlm.py`, `src/train_derma.py`, `src/run_config.py` |
| Image-only and LLaVA baselines | `src/baselines.py`, `src/llava_baselines.py`, `train.py` |
| Evaluation and statistical analysis | `src/eval_classify.py`, `src/eval_generate.py`, `src/stats.py` |
| Frozen case-level predictions and generated outputs | `src/outputs/` |
| Figures and LaTeX tables | `src/figures/`, `src/tables/` |
| Run configurations and step logs | `src/models/` |
| Environment specifications | `requirements.txt`, `environment.yml` |
| Original source metadata and attribution | `source_metadata/` |

The original dermoscopic images and trained model weights are **not included**. The repository contains derived data, source metadata, code, frozen predictions, generated outputs, and analysis artifacts.

## Study snapshot

- Source collection: DERM12345 dermoscopic dataset.
- Selected cohort: 500 images.
- Image-level split: 300 training, 74 validation, and 126 test images.
- Binary test endpoint: 118 images after excluding 8 indeterminate test images.
- Structured attributes: 11 deterministic image-derived proxies.
- Primary configuration: image-predicted attributes supplied to the fusion module.
- Reported primary test result: AUROC 0.963 and AUPRC 0.918.

The split is image-level because the selected 500-image subset does not retain a reliable mapping to the source patient and lesion identifiers. Repeated-patient, repeated-lesion, or near-duplicate leakage therefore cannot be excluded.

## Data source and attribution

The source collection is:

> Yilmaz, A., Yasar, S. P., Gencoglan, G., and Temelkuran, B. **DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses.** *Scientific Data* 11, 1302 (2024). DOI: `10.1038/s41597-024-04104-3`.

Dataset record: `10.7910/DVN/DAXZ7P`.

DERM12345 was released under the Creative Commons Attribution 4.0 International license. The complete attribution and licensing notes are provided in [`NOTICE`](NOTICE), [`DATA_LICENSE.md`](DATA_LICENSE.md), and `source_metadata/`.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── DATA_LICENSE.md
├── CITATION.cff
├── DATA_CARD.md
├── MODEL_CARD.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── environment.yml
├── data.jsonl
├── metadata.csv
├── images/
├── source_metadata/
├── prior_baseline_eval/
├── llava-base-model/
├── llava-finetuned/
├── train.py
└── src/
    ├── attributes/
    ├── data/
    ├── figures/
    ├── models/
    ├── outputs/
    ├── tables/
    └── *.py
```

## Verify the frozen data package

```bash
cd src/data
sha256sum -c checksums.sha256
cd ../..
```

The released checksums cover the frozen split manifest and the three split Q&A files.

## Environment setup

The reported runs used Python 3.11, PyTorch 2.9, CUDA 12.8, Transformers, PEFT, and bitsandbytes.

### Option 1: Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate dermaattr-vlm
```

A CUDA-capable Linux system is required for the reported 4-bit QLoRA training workflow. Ensure that the NVIDIA driver is compatible with the CUDA version used by the installed PyTorch build.

## Base model and image setup

Download the base model into `llava-base-model/`:

```bash
hf download llava-hf/llava-1.5-7b-hf --local-dir llava-base-model
```

Place the 500 source images in `images/` using the local names listed in `src/data/split_manifest.csv` (`001.jpg` through `500.jpg`). The original source-to-local mapping is not included and cannot be reconstructed reliably from this package alone.

## Reproduce tables and figures from frozen outputs

The reported statistical analyses can be regenerated without the source images or model weights:

```bash
python src/stats.py full_s0 resnet50
python src/make_figures.py
python src/make_extra_figures.py
```

Generated artifacts are written to `src/outputs/`, `src/figures/`, and `src/tables/`.

## Train and evaluate DermaAttr-VLM

```bash
python src/train_derma.py \
  --config full \
  --epochs 3 \
  --seed 0 \
  --per_image_cap 6 \
  --accum 8 \
  --out src/models/full_s0

python src/eval_classify.py \
  --model src/models/full_s0 \
  --config full \
  --seed 0

python src/eval_generate.py \
  --model src/models/full_s0 \
  --config full \
  --seed 0 \
  --n 250
```

Available configurations are:

- `full`: image-predicted attributes;
- `oracle`: reference attributes supplied at inference, for diagnostic analysis only;
- `shuffled`: fixed dataset-level shuffled-attribute control;
- `no_attr`: no structured attribute tokens.

The oracle condition is not deployable. The shuffled run listed in `src/outputs/invalidated_outputs.json` must be regenerated before it is used in analysis.

## Reconstruct derived data

After source images and authoritative source identifiers have been restored:

```bash
python src/audit_dataset.py \
  --manifest metadata.csv \
  --out src/data/pre_split_audit.json

python src/freeze_splits.py
python src/audit_dataset.py
python src/extract_attributes.py
```

Keep a separate copy of the released frozen split files before generating a new split.

## Important limitations

- The selected subset does not retain a reliable source-image, patient, or lesion mapping.
- The train/validation/test assignment and bootstrap intervals are image-level.
- The attributes are algorithmic proxies, not dermatologist annotations.
- The Q&A corpus contains templated language and repeated question forms.
- Generation metrics measure lexical agreement and do not establish explanation fidelity.
- The reported component analysis is based on one seed.
- External clinical validation has not been performed.

See [`DATA_CARD.md`](DATA_CARD.md) and [`MODEL_CARD.md`](MODEL_CARD.md) for the complete intended-use and limitation statements.

## Citation

The paper is currently submitted. Until a journal DOI is assigned, cite the software release using [`CITATION.cff`](CITATION.cff) and cite the DERM12345 source dataset separately.

```bibtex
@software{muhammad2026dermaattrvlm,
  title        = {DermaAttr-VLM: Code and Derived Data},
  author       = {Muhammad, Abdul Malik and Hussain, Tassadaq and Rehman, Muhammad ZiaUr and Alharbi, Soltan and Tan, Kim Geok},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/ucerd/llava_skin_cancer}
}
```

## Licenses

- Source code: [MIT License](LICENSE).
- Author-generated derived data and documentation: [CC BY 4.0](DATA_LICENSE.md).
- DERM12345 source metadata and any source images remain subject to the original DERM12345 CC BY 4.0 terms and attribution requirements.
- Third-party models and libraries retain their own licenses and terms.

## Contact

Questions about the repository may be submitted through GitHub Issues. Scientific correspondence should use the corresponding-author details given in the manuscript.
