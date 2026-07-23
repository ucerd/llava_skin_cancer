# Model Card: DermaAttr-VLM

## Model description

DermaAttr-VLM is a research vision-language model for dermoscopic image analysis. The implementation uses:

- a CLIP ViT-L/14-336 visual encoder;
- 576 visual patch tokens in a 1024-dimensional visual space;
- 11 normalized dermoscopic attribute proxies;
- four clinically motivated attribute groups and eight attribute tokens;
- gated visual-to-attribute cross-attention;
- projection into the LLaVA language-model space;
- a LLaVA-1.5-7B/Vicuna decoder adapted with LoRA/QLoRA;
- auxiliary malignancy-classification and attribute-reconstruction heads.

## Primary configuration

The primary deployable configuration predicts the attribute vector from the image and passes the predicted vector to the fusion module. Reference attributes are used as supervision. The `oracle` configuration supplies the reference attributes and is not deployable.

## Training configuration

- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.05.
- Optimizer: AdamW.
- Base learning rate: `2e-5`.
- Weight decay: `1e-2`.
- Gradient clipping: 1.0.
- Training epochs: 3.
- Effective batch size: 8.
- Maximum Q&A pairs per image: 6.
- Optimizer updates: 672.
- Reported hardware: one NVIDIA RTX 4070 Ti 12 GB per configuration.

## Evaluation

The primary binary endpoint was malignant versus benign classification on 118 held-out images.

| Metric | Primary result |
|---|---:|
| AUROC | 0.963 |
| AUPRC | 0.918 |
| Accuracy | 0.890 |
| Sensitivity | 0.933 |
| Specificity | 0.875 |
| Brier score | 0.067 |
| ECE | 0.053 |

The no-attribute configuration achieved AUROC 0.959. The reported AUROC difference was not statistically significant. The study therefore does not establish a separate attribute-content effect.

## Intended use

The model is intended for research on multimodal medical-image representation, structured attribute conditioning, reproducibility, and internal benchmarking.

## Prohibited and inappropriate use

The model is not intended for:

- clinical diagnosis;
- autonomous screening or triage;
- treatment or management recommendations;
- patient-facing deployment;
- claims of explanation fidelity;
- claims of generalization across devices, sites, or demographic groups.

## Limitations

- Internal image-level split only.
- No reliable patient- or lesion-level identifiers for the selected subset.
- Small held-out sample.
- No external validation.
- Algorithmic attributes rather than dermatologist annotations.
- Templated Q&A corpus.
- Generation metrics do not measure clinical grounding.
- Single-seed component analysis.
- Trained adapter and custom-head weight files are not included in this release.

## Ethical considerations

Outputs can contain incorrect subtype labels or unsupported clinical wording. Any generated response must be treated as experimental model output rather than medical information. Human expert review and independent validation are required before any clinical investigation.
