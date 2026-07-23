# Data Card: DermaAttr-VLM Derived Data Package

## 1. Summary

This repository contains the derived data package used for the DermaAttr-VLM internal retrospective study. It includes a frozen image-level split manifest, template-based question-answer pairs, deterministic dermoscopic attribute proxies, case-level predictions, generated responses, and evaluation artifacts.

The original dermoscopic images are not included.

## 2. Source dataset

The selected images originate from DERM12345:

- Yilmaz et al., *Scientific Data* 11, 1302 (2024).
- Article DOI: `10.1038/s41597-024-04104-3`.
- Dataset DOI: `10.7910/DVN/DAXZ7P`.
- Upstream license: CC BY 4.0.

The included full source metadata file contains 12,345 dermoscopic records and 1,627 unique patient identifiers. The 500-image study subset, however, uses local numeric identifiers and does not retain a reliable mapping to those authoritative source records.

## 3. Cohort composition

| Clinical group | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Benign | 350 | 210 | 52 | 88 |
| Malignant | 120 | 72 | 18 | 30 |
| Indeterminate | 30 | 18 | 4 | 8 |
| **Total** | **500** | **300** | **74** | **126** |

The binary test endpoint contains 118 images after excluding the eight indeterminate test images.

## 4. Files

- `metadata.csv`: local 500-image metadata used by the split-generation code.
- `data.jsonl`: unsplit image-linked Q&A corpus.
- `src/data/split_manifest.csv`: frozen local image-level split.
- `src/data/qa_train.jsonl`: 5,999 Q&A records.
- `src/data/qa_val.jsonl`: 1,481 Q&A records.
- `src/data/qa_test.jsonl`: 2,520 Q&A records.
- `src/attributes/attributes.csv`: 11 normalized image-derived attribute proxies for 500 images.
- `src/outputs/`: case-level scores, thresholds, generated responses, and summary statistics.

## 5. Structured attributes

The attribute vector contains:

1. asymmetry;
2. border irregularity;
3. colour variegation;
4. number of visible colours;
5. pigment network;
6. dots/globules;
7. streaks;
8. blue-white veil;
9. regression structures;
10. vascular patterns; and
11. ulceration/crusting.

These values are deterministic image-derived proxies. They are **not expert annotations** and may contain substantial measurement error.

## 6. Q&A corpus

The corpus contains approximately 20 Q&A pairs per image and uses structured, template-based phrasing. Many test questions repeat forms seen during training, and some prompts contain only an image token. Generated-answer metrics should therefore be interpreted as lexical agreement with reference wording, not as evidence of image grounding, clinical reasoning, or explanation fidelity.

## 7. Intended uses

Appropriate uses include:

- reproducing the DermaAttr-VLM internal analyses;
- studying attribute-conditioned multimodal architectures;
- auditing the released split, predictions, and statistical procedures;
- benchmarking research code under the stated image-level split.

## 8. Out-of-scope uses

The package must not be used for:

- clinical diagnosis or treatment decisions;
- patient triage;
- claims of dermatologist-level performance;
- deployment without independent validation;
- demographic or fairness claims unsupported by the available metadata.

## 9. Known limitations

- No reliable source-image, patient, or lesion mapping for the selected subset.
- Image-level rather than patient- or lesion-level splitting.
- Possible repeated-patient, repeated-lesion, or near-duplicate leakage.
- One public source collection and no external cohort.
- Incomplete acquisition-device, site, and skin-tone metadata.
- Templated Q&A references.
- Algorithmic attribute proxies rather than expert labels.

## 10. Licensing

Author-generated derived data are released under CC BY 4.0. Upstream DERM12345 material remains subject to its original CC BY 4.0 terms. See `DATA_LICENSE.md` and `NOTICE`.
