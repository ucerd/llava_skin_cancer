# Maintainer Release Checklist

1. Confirm that no original images or restricted weights are staged.
2. Confirm that no tokens, passwords, private paths, or patient-identifiable fields are present.
3. Run `python -m compileall -q .`.
4. Run `cd src/data && sha256sum -c checksums.sha256`.
5. Confirm that `README.md`, `DATA_CARD.md`, `MODEL_CARD.md`, and the manuscript use the same cohort and metric values.
6. Update `CITATION.cff` when a journal DOI is assigned.
7. Create a GitHub release named `v1.0.0`.
8. Archive the release in Zenodo and add the Zenodo DOI badge and citation.
9. Add repository description, website, and topics in GitHub settings.
10. Verify the public repository from a signed-out browser session.
