# Contributing

Contributions that improve reproducibility, documentation, testing, or analysis are welcome.

## Before opening a pull request

1. Do not add patient-identifiable information.
2. Do not upload source dermoscopic images or restricted model weights without verifying redistribution rights.
3. Keep released frozen results separate from newly generated experimental outputs.
4. Run:

```bash
python -m compileall -q .
cd src/data && sha256sum -c checksums.sha256
```

5. Describe any change that affects splits, preprocessing, attributes, thresholds, or reported metrics.

Scientific changes should include an explanation of their effect on reproducibility and manuscript consistency.
