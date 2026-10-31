# Final fusion protocol

Protocol: `vida-eight-family-ms-v2-20261030`.

The branch-development resource contains 32,000 original videos. Its historical
split labels do not define the independent final evaluation used here. The
16,400-video suite has 5,120 fusion-training, 640 probability-calibration,
640 threshold-selection, 8,000 in-domain and 2,000 ModelScope-test originals.
Known source/prompt groups separate the roles. The new originals exclude the
base collection and previously evaluated video identities. Unknown complete
parent/prompt mappings prevent claiming a proof against all semantic duplicates.

Only fusion-training labels select the MLP/LR candidate and training duration.
ModelScope generated videos are absent from branch development, fusion fitting,
probability calibration and threshold selection in the declared project data.
The real controls come from the known Vript/HD-VG sources. This is a generated-
family holdout rather than an assertion that both real and generated sources
are new. Earlier ModelScope results had been observed, so this independent-
sample confirmation does not constitute the first blind encounter with that
generator. Historical Sora/Kling results remain in the full results report.

## Fixed branch lineage

| Branch | Artifact used |
|---|---|
| A | Partial CLIP Round 4, checkpoint SHA256 `12d7c3898482ada16b45f99832c9e95914388eec8c7f07bd5184ef554b86b2e1` |
| B | PR #2 LightGBM artifact SHA256 `66e13e1e787c12e88dd9b66d9d789d6654e117e3a9a59db77d25c83cba38a646` |
| C | Real-reference SVM nu 0.01, model ID `5649461f74586745570f74b2b0db77222780391c12c734feeb973ec743d6ddc6` |

B fitted the full 32K collection. A uses its delivered partial-tuning checkpoint,
and C's estimator fits real videos only. The common resource does not imply
identical actual fitting scopes. A has no complete published per-video fitting
ID list, and cross-environment reproduction of historical A scores has a recorded
difference. All new-suite A predictions use one fixed local runtime and model.

## Branch dependencies

A implementation is in the current upstream `branches_dev/branch_a/dev` directory. Its
selected checkpoint is tracked through upstream LFS, while its exact pretrained
CLIP base is obtained separately. B's sealed implementation/model is in
[PR #2](https://github.com/Jerry00C/Vida/pull/2), expected at
`branches_dev/branch_b`. C implementation is in
[PR #1](https://github.com/Jerry00C/Vida/pull/1), expected at `branches_dev/branch_c`.
C weights are in the upstream `branch-c-artifacts-2026-10-29` release.
These independent branch deliveries are prerequisites for raw-video adapters;
they are not needed for the released fusion model when A/B/C scores are supplied.

## Format control

The paired 2,000-video ModelScope control changes duration support, sampling,
geometry and encoding together. All branch weights, fusion weights, calibration
and the threshold remain fixed. These are copies of the same originals,
not additional independent observations. The comparison cannot isolate one
causal factor such as FPS. It is not used to fit a replacement threshold.
