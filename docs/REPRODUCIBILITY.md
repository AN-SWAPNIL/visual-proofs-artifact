# Reproducibility and Scope

## What Is Preserved

The six notebooks in `codes/` and all selected files in `results/` are exact
copies from the tracked accepted-study workspace. Their comments and inline
outputs are retained. `provenance.json` records the source revision, original
relative paths, byte counts, and SHA-256 checksums. No new model experiments
or revised predictions have been substituted.

The randomized generator is the tracked implementation at the source revision.
The separate `non_randomized/` implementation is recovered from commit
`58c2125` before the randomization module was introduced. It covers the same
eight properties and three categories, but its identity as the exact source
of every archived non-randomized image is not independently established.
The supplied archive/content checksums are authoritative for exact images.

Excluded material includes later untracked `_all.ipynb` notebooks; new model
families and prompt-control experiments; incomplete thesis extensions;
credentials; weights/checkpoints; Kaggle cache manifests and duplicate
`__results___files` image exports; and `cv_results.json`,
`all_models_comparison.png`, and `presentation_summary.txt`, which are not
saved outputs of the included vision notebook. Original files in the
research workspace were not deleted.

## Prompts and VPG

The archived evaluation function constructs a prompt with the property name,
a YES visual rule, a pipeline-specific negative visual rule, and a request
for `YES/NO: [brief reason]`. The raw prediction CSVs store the actual prompts.
The no-demonstration setting is zero-shot **with supplied property/visual
instructions**, not instruction-free property recognition.

The negative rule differs between Exp1 and Exp2, including for the positive
images. Consequently VPG combines differences in the negative collections,
layout, graph construction, and supplied instructions. It is not a causal
estimate of certificate reasoning alone. The original prediction parser is
also preserved, including fallback choices; no modern parser is silently
applied to change the published numbers.

Precision, recall, and F1 in the saved MLLM summaries treat YES as positive.
For example, high precision describes correctness among YES predictions,
not NO predictions. Differences in accuracy should be reported as
**percentage points**. The comparison of Qwen2.5-VL-3B and Qwen3-VL-8B also
changes model generation, so it does not isolate parameter count.

## Dataset Provenance Caveat

The artifact labeled `mllm_model_evaluation_3_wr` is the historical run
reported as non-randomized. Its original notebook configuration and saved
prediction paths use `graph-visualization`, not the separate
`graph-visualization-wr` path. A historical Kaggle dataset version or mount
may explain this, but the stored path alone cannot prove it.

There is supporting evidence for the reported condition: the notebook's
embedded manual-test input, `hamiltonian/yes/hamiltonian_yes_001.png`, has
exactly the same dimensions and RGB pixel hash as the `wr` archive image,
and does not match `r`. The other three zero-shot notebooks' manual examples
also uniquely match their intended collections. Run
`python scripts/check_input_examples.py` after installing Pillow, with both
reference archives available, to repeat this check. The recorded comparisons
are in `docs/input_example_checks.json`.

This check supports the intended condition but verifies one displayed input
per notebook, not all evaluation calls or a complete historical input version.
Recover full input/version records before claiming complete provenance.
The original labels and files are preserved, rather than relabeled or rewritten.

The runnable-copy helper requires an explicit dataset path for any NEW run;
this does not retroactively establish the input to the archived run.

## Splits

Each pipeline has 16,000 input images and a 70%/15%/15% train/validation/test
split (11,200/2,400/2,400 images), with `random_state=42`.

- MLLM fine-tuning: samples are loaded in sorted filename order and split
  using `train_test_split`, stratified by the combined property/category
  key. The first split holds out 30%, and the second divides that held-out
  part equally into validation and test. Each property has 300 test images.
- Vision training: samples are loaded with `os.listdir` and split using the
  same two-stage method, but stratified by the binary label only. Test counts
  therefore vary by property (290--313 in the supplied summaries), rather
  than being exactly 300 for each property. Directory iteration order can
  affect the exact selected files across filesystems.
- Zero-shot evaluation: the original loader takes up to the first 200
  directory entries in each property/category using `os.listdir`. This is
  3,200 evaluations per pipeline. It is not the supervised held-out test
  split, and the cap is not a seeded random sample. Saved prediction CSVs
  identify the evaluated filenames.

These are image/sample-level splits. There is no verified graph-isomorphism
deduplication, cross-view graph-ID grouping, or disjoint generator-seed split
in the released implementation. A fixed split seed is not evidence of
graph-level independence. Augmented views are applied after dataset splitting,
but that does not establish independence of related source graphs. Do not
claim stronger safeguards than the available records demonstrate.

## Hardware and Dependencies

The saved zero-shot logs identify Tesla T4 and PyTorch `2.8.0+cu126`.
They log device 0 and do not by themselves establish the number of GPUs used.
The MLLM fine-tuning and vision notebooks identify NVIDIA GeForce RTX 4070 Ti
and PyTorch `2.6.0+cu124`; the vision log reports approximately 12.9 decimal GB.
These logs should be reconciled with any manuscript statement that all
training used Kaggle T4 sessions. GPU memory across two devices is not a
single automatically pooled allocation.

The original zero-shot install cell fetches moving Transformers Git main;
the vision setup tries to install NumPy 1.24.4. No complete historical
dependency lockfile or model commit IDs were recovered. `requirements/`
provides explicit bounded starting dependencies for a new Python 3.11
environment, not a claim of exact environment recovery. The new-run helper
uses these requirements, retaining the original installer text as comments.
It otherwise changes dataset/output/checkpoint paths only, clears old run
outputs, and leaves model and experiment code intact.

Generation also uses Python's built-in `hash(property_name)` in its per-image
seed expression. Without the historical `PYTHONHASHSEED`, a CLI seed alone
does not guarantee identical regenerated images across processes. For NEW
generation record a fixed hash seed in addition to package versions, but do
not claim that this reconstructs the old data. The non-randomized and
randomized collections also change graph-generation settings such as node
count ranges; they are not verified renderings of an identical graph set.
All archived randomized PNGs are 400x400; all non-randomized PNGs are
406x406. Their metadata reports node-count ranges of 10--60 and 14--40,
respectively. These observations should be distinguished from nominal
canvas settings and configured node ranges in manuscript descriptions.

## What the Checks Establish

`scripts/check_results.py` verifies original file checksums and recomputes
MLLM aggregate/per-property accuracy and F1 (plus aggregate precision and
recall) from CSVs. It checks the eight vision backbone/pipeline overall
accuracies against their property confusion matrices. The two ensemble
summaries have no saved per-property matrices or raw per-image predictions;
their metrics are retained but cannot be independently recomputed here.

The tests check helper behavior, safe extraction, and notebook preparation.
They do not validate the mathematical truth of every generated graph label,
rerun CUDA models, prove absence of shortcuts/leakage, or establish that
attention maps demonstrate structural reasoning. Grad-CAM is qualitative
localization evidence, not a causal test of what a model uses.
