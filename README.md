# Learning from Visual Proofs

Research artifacts for the following paper, accepted at **ICAA 2027**.

## Paper

**Title:** Learning from Visual Proofs: Neural Models for Network Property Verification

**Authors:** Most. Sonia Khatun\*, Ahmmad Nur Swapnil\*, Reyan Ahmed, and Md. Saidur Rahman.

\* These authors contributed equally to this work and share first authorship.

**Conference:** International Conference on Applied Algorithms (ICAA 2027).

The paper studies whether graph drawings that present visual certificates help
neural models recognize structural graph properties. It introduces a
24,000-image dataset covering eight properties and compares same-style
negative examples with explicit counter-certificate layouts. The evaluation
examines zero-shot MLLMs, an MLLM after fine-tuning, and supervised vision
models to distinguish recognition without task-specific training from
learnability with supervision.

This repository contains the accepted study only. The later multi-model
revision experiments, new prompt ablations, and incomplete thesis extensions
are not included. The six original notebooks retain their code, comments,
execution logs, and inline figures. `provenance.json` records their origin and
the saved output files selected for this release.

## Study

Eight graph properties are represented by `yes` (the property holds), `s_no`
(the property fails in the positive layout style), and `d_no` (the property
fails with a different, counter-certificate layout). Exp1 compares `yes`
with `s_no`; Exp2 compares `yes` with `d_no`. Visual Proof Gain (VPG) is
Exp2 accuracy minus Exp1 accuracy, expressed in percentage points.

**VPG measures a difference between these experimental conditions. It does
not isolate certificate reasoning from layout or prompt effects.** The saved
zero-shot runs use property-specific rules and request a short reason, without
task-specific demonstration examples. The negative rule changes between
Exp1 and Exp2. See [reproducibility notes](docs/REPRODUCIBILITY.md) before
interpreting the results.

## Contents

| Path | Contents |
| --- | --- |
| `data_generator/` | Randomized generator, eight property modules, randomization and reasoning modules |
| `data_generator/non_randomized/` | Pre-randomization generator snapshot recovered from Git history |
| `codes/` | Six original, executed notebooks, preserved byte-for-byte |
| `results/` | CSV predictions, JSON summaries, and figures saved by those notebooks |
| `datasets/` | Dataset catalog, per-file checksums, license, and acquisition instructions |
| `requirements/` | Separate dependencies for generation, zero-shot evaluation, MLLM fine-tuning, and vision training |
| `scripts/` | Dataset verification/extraction, runnable notebook preparation, and saved-result checking |
| `tests/` | CPU-only tests of the release helpers |

The original `fintetune` spelling in two result directory names is retained
to avoid silently renaming archived artifacts. Later `_all.ipynb` notebooks,
model weights, caches, duplicate Kaggle HTML image exports, and unrelated
cross-validation/presentation outputs are excluded.

## Datasets

Each condition contains **24,000 PNG images**, with 1,000 images per
property/category. `r` means randomized; `wr` means **without randomization**,
not merely without node-order randomization. They are separate collections;
matching filename numbers do not establish that two images share a graph.
The supplied randomized PNGs are 400x400 pixels; the non-randomized PNGs are
406x406 pixels. Their metadata reports node-count ranges of 10--60 and
14--40, respectively.

| Condition | Dataset | Local archive |
| --- | --- | --- |
| `r` | [Kaggle randomized dataset](https://www.kaggle.com/datasets/ahmmadnurswapnil/graph-visualization) | `datasets/archives/graphtrails_dataset.zip` |
| `wr` | [Kaggle non-randomized dataset](https://www.kaggle.com/datasets/ahmmadnurswapnil/graph-visualization-wr) | `datasets/archives/graphtrails_dataset_wr.zip` |

The Kaggle addresses come from archived input paths. Authors must confirm
public access, exact versions, licensing, and content checksums before
announcing the release. Local copies of both archives are supplied in this
prepared workspace but are intentionally ignored by Git. A Git clone obtains
the data from Kaggle or separately uploaded release assets, not from Git
history. See [dataset instructions](datasets/README.md).

## Setup

1. Use a separate Python 3.11 environment for the selected workflow.
   The original runs used Python 3.12, but the original vision installer pins
   an incompatible older NumPy; the runnable-copy helper replaces that installer.
2. For GPU work, first install a compatible CUDA-enabled `torch` and
   `torchvision` pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
   Do not combine incompatible releases. Historical logs show PyTorch 2.8.0
   for zero-shot runs and 2.6.0 for the two training notebooks.
3. Install only the relevant requirements file:

```sh
python -m pip install -r requirements/generation.txt
# Or choose ONE of these for a model workflow:
python -m pip install -r requirements/zero-shot.txt
python -m pip install -r requirements/mllm-finetune.txt
python -m pip install -r requirements/vision.txt
```

These requirements are bounded starting environments, not exact historical
lockfiles. CUDA inference/training has not been re-executed as part of packaging.
Record `pip freeze`, model revision, dataset checksums, and hardware for a new run.

## Inspect or Verify Saved Results

No GPU or dataset download is needed for this step:

```sh
python scripts/check_results.py
python -m unittest discover -s tests
```

The first command checks the five MLLM result groups against their prediction
CSVs and the vision summary against its property confusion matrices. It does
not run a model or prove graph-level independence.

## Run the Notebooks

Each zero-shot notebook runs both Exp1 and Exp2 for its own model/condition.
The MLLM fine-tuning notebook trains and evaluates two separate adapters.
The vision notebook trains the four reported backbones and their ensemble.
**Zero-shot evaluation and fine-tuning remain separate workflows.**

| Notebook stem (under `codes/`) | Workflow | Data |
| --- | --- | --- |
| `mllm_model_evaluation_2.5` | Qwen2.5-VL-3B zero-shot | `r` |
| `mllm_model_evaluation_2.5_wr` | Qwen2.5-VL-3B zero-shot | `wr` |
| `mllm_model_evaluation_3` | Qwen3-VL-8B zero-shot | `r` |
| `mllm_model_evaluation_3_wr` | Qwen3-VL-8B zero-shot, reported non-randomized run; provenance caveat applies | `wr` intended |
| `mllm_model_finetune` | Qwen2.5-VL-3B QLoRA fine-tuning and held-out evaluation | `r` |
| `vision_model_finetune` | Supervised vision training and held-out evaluation | `r` |

1. Verify and extract the selected dataset, as described in `datasets/README.md`.
2. Create a fresh runnable copy, for example:

```sh
python scripts/prepare_notebook.py mllm_model_evaluation_3 --dataset-root datasets/extracted/r/graphtrails_dataset
python -m jupyterlab
```

3. Open the printed `runs/<notebook>/run.ipynb`, select the prepared environment,
   and run cells in order. The first cell changes to the repository root. Outputs
   and checkpoints go under `runs/`, leaving the archived evidence unchanged.
4. For Kaggle, upload/clone this repository and attach the correct dataset and
   model. Use its actual mounted dataset path as `--dataset-root`. The helper
   changes paths and dependency setup only; prompts, model identities, sample
   caps, training settings, and parsers remain the original ones.

The helper refuses to overwrite a previous run. Use `--run-name` for a new
directory. For the `_3_wr` notebook it makes the intended dataset selection
explicit. Its archived manual-test image matches `wr`, but its recorded path
does not independently identify the full historical input version.

## Generate New Images

```sh
python data_generator/graphtrails_dataset_generator.py --property all --n_per_class 1000 --seed 42 --output_dir generated/r
python data_generator/non_randomized/graphtrails_dataset_generator.py --property all --n_per_class 1000 --seed 42 --output_dir generated/wr
```

Use `--n_per_class 1` for a smoke test. All eight properties and three categories
give `8 * 3 * n_per_class` images. Some legacy help text incorrectly says two
categories or nine properties; the released code is preserved rather than
silently edited. Generation can be slow for some properties. The historical
snapshot is a recovered implementation, not proof of the precise source used
to produce every archived image. Use the supplied dataset checksums for exact
published image recovery. Python hashing and library versions also affect new
randomized generation; see the reproducibility notes.

## Paper, Citation, and Licenses

The paper title and authors are recorded in `CITATION.cff`. DOI, volume, and
page numbers should be added when available; none are
invented here. The LaTeX production bundle, publisher PDF, reviews, registration
receipt, and signed publishing agreement are **not** part of this code release.
A paper link is sufficient; deposit an author manuscript only after checking
the applicable publishing agreement and sharing policy.

Code and accompanying software documentation: [MIT](LICENSE).
Graph images, labels, and reasoning texts: [CC BY 4.0](datasets/LICENSE.md).
Model-generated responses retain any applicable model/provider terms. Model
weights and third-party software are not relicensed. Coauthors should approve
the release and dataset license before public publication.
