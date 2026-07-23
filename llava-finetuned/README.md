# Standard LLaVA-LoRA adapter

The standard LLaVA-LoRA baseline adapter weights are not included in this source/data
package. Run `python train.py` from the repository root to regenerate them here. The
adapter configuration is retained in `adapter_config.json`, and the frozen case-level
scores used in the paper are available in `../src/outputs/`.
