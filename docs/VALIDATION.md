# Local Packaging Validation

Checked on November 3, 2026. These are artifact-packaging checks, not new
paper experiments or a rerun of model training/inference.

| Check | Outcome |
| --- | --- |
| Original file provenance | 71 copied/recovered source, notebook, and result files match recorded SHA-256 values |
| Notebook preservation | All six original notebooks retain their source/comments and inline outputs byte-for-byte |
| Notebook validity | Original notebook JSON/nbformat validation passed; original and prepared code passed syntax validation (264 code cells in total) |
| Release helper unit tests | Nine CPU-only tests passed |
| MLLM result verification | All five result groups, both pipelines each, agree with their saved prediction CSVs for aggregate metrics and per-property accuracy/F1 |
| Vision backbone verification | Eight backbone/pipeline overall accuracies agree with their saved property confusion matrices |
| Vision ensemble limitation | Only aggregate summaries are available; no independent recomputation from per-image outputs was possible |
| Randomized archive | All 24,002 image/metadata files verified; archive SHA-256 matches |
| Non-randomized archive | All 24,001 image/metadata files verified; archive SHA-256 matches |
| Image counts | Each collection has 24 groups with 1,000 PNGs per group |
| PNG dimensions | All 24,000 `r` PNGs are 400x400; all 24,000 `wr` PNGs are 406x406 |
| Archived manual input examples | All four zero-shot manual-test images uniquely match their intended collection by RGB pixel hash and dimensions; this is not full-run input verification |
| Randomized generation smoke test | One image per category/property produced all 24 expected PNGs and metadata/reasoning files |
| Historical non-randomized generation smoke test | One image per category/property produced all 24 expected PNGs and metadata |
| Runnable notebook preparation | Separate zero-shot and fine-tuning copies created successfully without changing archived notebooks |

The smoke tests used the available host environment: Python 3.14.2,
NumPy 2.4.1, NetworkX 3.6.1, Matplotlib 3.10.8, and SciPy 1.17.1.
The proposed Python 3.11 GPU requirements were not installed or GPU-tested
during packaging. No exact historical environment lockfile is claimed.

The requirements bounds are starting configurations for new runs; verify
model-library/CUDA compatibility in the intended environment. No models,
datasets, or paperwork were uploaded to GitHub, Kaggle, CMT, or email by this
task. Public dataset versions and license metadata remain author checks.
