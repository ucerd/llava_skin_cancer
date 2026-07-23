# Data Dictionary

## `metadata.csv`

This is the local 500-image metadata file used by the split-generation scripts.

| Column | Meaning |
|---|---|
| `isic_id` | Local numeric study identifier; it is not the authoritative ISIC identifier. |
| `anatom_site_general` | General anatomic location when available. |
| `anatom_site_special` | More specific anatomic location when available. |
| `diagnosis_1` | Top-level label: Benign, Malignant, or Indeterminate. |
| `diagnosis_2`–`diagnosis_5` | Hierarchical diagnostic labels when available. |
| `diagnosis_confirm_type` | Diagnostic confirmation type when available. |
| `melanocytic` | Boolean indicator for melanocytic lineage. |
| `Unnamed: 9` | Empty legacy export column retained to preserve the frozen source file. |

## `src/data/split_manifest.csv`

| Column | Meaning |
|---|---|
| `isic_id` | Local numeric study identifier. |
| `image` | Local filename in `images/`. |
| `diagnosis_1` | Top-level clinical group. |
| `y` | Binary endpoint: 1 malignant, 0 benign, -1 indeterminate. |
| `confirm` | Confirmation type or `not_reported`. |
| `split` | Frozen image-level split: train, val, or test. |
| `melanocytic` | Boolean melanocytic indicator. |

## `src/attributes/attributes.csv`

All attribute columns are normalized to `[0, 1]` and are deterministic image-derived proxies, not expert annotations.

| Column | Computation basis |
|---|---|
| `asymmetry` | Principal-axis area mismatch of the segmented lesion mask. |
| `border_irregularity` | Boundary compactness. |
| `color_variegation` | Chroma variance in Lab space within the lesion. |
| `num_colors` | Count of clinical palette colours above the occupancy threshold. |
| `pigment_network` | Mid-frequency edge energy inside the lesion. |
| `dots_globules` | Circular-blob density. |
| `streaks` | Peripheral-ring edge density. |
| `blue_white_veil` | Fraction of bluish-white pixels. |
| `regression_structures` | Fraction of scar-like high-lightness, low-chroma pixels. |
| `vascular_patterns` | Fraction of red-dominant pixels. |
| `ulceration_crusting` | Fraction of yellow-red high-intensity crust-like pixels. |

## Q&A JSONL files

Each line contains:

- `id`: question-answer record identifier;
- `image`: local image filename;
- `conversations`: two-turn list containing a human prompt and a reference answer.
