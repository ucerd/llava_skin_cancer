# Model run metadata

Each run directory retains its experiment configuration, LoRA adapter configuration,
and per-step training log. Large trained adapter and custom-head weight files are not
included in this source/data release. They can be regenerated with `train_derma.py`
or `run_config.py`; the frozen case-level predictions used for the paper are retained
in `../outputs/`.
