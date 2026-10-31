# Reproduce fusion fitting

Run commands from the repository root. Keep raw videos, data manifests and
generated runs outside tracked source. The released model can be used directly
without reproducing training.

## Environment

Fusion code uses Python 3.12 and `requirements.lock.txt`. Epoch fitting runs in
a separate environment with `requirements.neural.lock.txt` (CPU PyTorch).
Configure the actual path of that environment's Python. Exporting branch
scores uses each branch's own pinned environment. Do not use the eight sparse
Branch A frames as a substitute for B/C temporal preprocessing.

## Inputs

1. Obtain the separately delivered original 32K manifest, the independent
   16,400-video suite, its original manifest and `DATA_VERIFICATION.json`, plus
   the historical suite manifest used for exclusion checks.
2. Obtain the exact fixed A/B/C checkpoints from their branch deliveries.
3. Use the export adapters or the original delivered native score files.
4. Copy `config.expanded.example.json` to a local configuration. Set paths to
   the data, score folder, a **new** output folder and the neural Python.

The expanded configuration resolves paths from the current working directory.
Use absolute paths or run from the repository root. The historical suite path
is an exclusion-audit dependency, not a source of final-test training samples.

```bash
python -m stacking.expanded --config stacking/config.expanded.local.json doctor
python -m stacking.expanded --config stacking/config.expanded.local.json \
  import-scores --branch A --input A_native.csv
python -m stacking.expanded --config stacking/config.expanded.local.json \
  import-scores --branch B --input B_native.csv
python -m stacking.expanded --config stacking/config.expanded.local.json \
  import-scores --branch C --input C_native.csv
python -m stacking.expanded --config stacking/config.expanded.local.json train
python -m stacking.expanded --config stacking/config.expanded.local.json evaluate
```

Score imports verify the complete video set, finite values, no abstentions,
media identities when present, and the declared branch checkpoint. Training
refuses to overwrite a frozen model. Use a new output directory for reproduction.
Final-test data do not select model type, duration, calibration or threshold.

The saved `family_cv_folds.json` and `training_report.json` record the actual
eight holdouts, epoch losses and selected candidate. The released final model
uses the median selected duration, 102 epochs. Exact numerical reproduction
also requires the recorded NumPy/sklearn/PyTorch versions and input scores.
