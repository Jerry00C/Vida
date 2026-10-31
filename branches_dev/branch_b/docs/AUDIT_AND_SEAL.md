# Acceptance audit and byte-exact SHA-256 seal

The acceptance receipt is `../DELIVERY_RECEIPT.json`; measured evidence is
`AUDIT_REPORT.json`. Receipt counts and byte size include **every regular file**,
including the receipt and both seal files. No symlinks or runtime caches belong
in the sealed package.

## Coverage and trust boundary

All paths in the seals are relative to `deliverables/branch_b_temporal_wavelet/`:

- `CHECKSUMS.txt`: SHA-256 of every file except the two seal files.
- `MANIFEST_SHA256.json`: SHA-256 and byte size of every file except itself,
  **including `CHECKSUMS.txt` and the acceptance receipt**.
- The final manifest's own SHA-256 is printed by the audit/verification command.
  Retain that digest independently with the handoff. A file cannot contain its
  own final checksum; the external digest completes the coverage without a
  circular self-hash. These are integrity checks, not authenticated signatures.

Treat the model joblib as executable trusted content: verify the trusted manifest
before loading it. Checksums bundled beside a model cannot authenticate a package
if an attacker can replace both the model and its manifests.

## Verification commands (repository root)

```bash
# Read-only: exact inventory, every digest, receipt count and final byte size.
python -B scripts/audit_temporal_wavelet_delivery.py --verify-only

# Standard text-manifest verification, run from the package directory.
(cd deliverables/branch_b_temporal_wavelet && sha256sum -c CHECKSUMS.txt)

# Full audit, fresh isolated notebook execution, and regeneration of the seal.
# This changes timestamps and notebook outputs; it intentionally reseals.
python -B scripts/audit_temporal_wavelet_delivery.py

# Independent regressions for receipt bookkeeping and tamper detection.
python -B -m pytest -q -p no:cacheprovider tests/test_temporal_wavelet_delivery_audit.py
```

Python dependencies and their measured versions are recorded in `AUDIT_REPORT.json`.
The full audit requires the generation-specific training feature parquet and label
metadata named in that report. The isolated 40-row API/CLI/notebook regression uses
only the copied deliverable and installed dependencies, not the training tree.
The Jupytext percent `.py` source is authoritative regardless of mtimes. The audit
regenerates via `--to ipynb --update` in a short staging directory to avoid Windows
MAX_PATH, executes with nbclient, and requires a repeated regeneration and actual
`--sync` to leave both files byte-identical. Notebook JSON is never hand-edited. Git attributes disable newline
normalization for this sealed package to preserve its exact hashes on checkout.

## Handoff scope

- The original model binary, thresholds, and full-dataset score values are unchanged.
- All 32,000 scores must reproduce **exactly**, with labels and families aligned
  by unique `clip_id`. The older merged feature cache is not the training input.
- The portable 40-row reference fixture is selected from the original saved
  predictions, not computed by the inference code under test.
- Independent canonical comparisons run in a clean separate subprocess: four
  decoded tensors, six independently decoded synthetic videos, and 20 available
  historical fit-role videos with media hashes checked against original receipts.
  All 113 features and margins must match exactly; canonical sklearn calibration
  allows only `atol=2e-16, rtol=0` float64 roundoff. Saved-score replay is exact.
  The original 32,000 source videos are **not** all replayed. Full-corpus
  regression uses generation-specific cached training features.
- Video-versus-feature API agreement is interface consistency, not an independent
  canonical extraction oracle. MPEG4 success and MJPG ABSTAIN both exercise the
  frozen allowed-codec policy; it is not relaxed to satisfy an outdated fixture.
- Hardened loader checks cover missing metadata/checksum/schema/calibration,
  corrupt state, thread/import coexistence, no permanent module shims, and
  standalone API/direct/module CLI from copied folders. The notebook kernel and
  integrated worker actively block imports of the original source namespaces.
- Requirements pins/imports are checked in the installed Windows/Python reference
  environment; no independent clean-venv install or cross-platform matrix is
  claimed. Historical evidence and original model/score payloads are preserved.
- These are full-fit **in-sample** scores for the requested internal stacking.
  `raw_margin` is the team's chosen interface. OOF generation or a holdout is
  not a prerequisite for this all-data handoff; these scores alone do not
  estimate generalization. See `PROVENANCE.md` for the directive change.
- The default per-source threshold meets 1% on each real source in this corpus.
  The secondary pooled threshold is **family-weighted** (1.0% weighted, 1.475%
  unweighted); it is not a 1% unweighted corpus guarantee.

Any edit, runtime cache, added file, or changed notebook output invalidates the
exact package inventory until the full audit and sealing process is repeated.
