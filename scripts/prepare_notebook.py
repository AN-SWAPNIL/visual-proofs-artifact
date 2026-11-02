"""Make a runnable copy without altering the original executed notebook."""

import argparse
import ast
import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "mllm_model_evaluation_2.5", "mllm_model_evaluation_2.5_wr",
    "mllm_model_evaluation_3", "mllm_model_evaluation_3_wr",
    "mllm_model_finetune", "vision_model_finetune",
]


def replace_config(source, replacements):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for assignment in node.body:
                if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
                    continue
                target = assignment.targets[0]
                if isinstance(target, ast.Name) and target.id in replacements:
                    value = assignment.value
                    # Only replace the value, retaining existing comments.
                    if value.lineno != value.end_lineno:
                        raise ValueError(f"Multiline config value: {target.id}")
                    changes.append((value.lineno - 1, value.col_offset,
                                    value.end_col_offset, repr(replacements[target.id])))
    for row, start, end, value in sorted(changes, reverse=True):
        encoded = lines[row].encode("utf-8")
        lines[row] = (encoded[:start] + value.encode("utf-8") + encoded[end:]).decode("utf-8")
    return "".join(lines), len(changes)


def make_copy(notebook, name, dataset, run_dir, artifact_root):
    result = copy.deepcopy(notebook)
    requirements = ("vision.txt" if name == "vision_model_finetune" else
                    "mllm-finetune.txt" if name == "mllm_model_finetune" else "zero-shot.txt")
    replacements = {
        "DATASET_ROOT": str(dataset), "RESULTS_DIR": str(run_dir / "outputs"),
        "CHECKPOINT_DIR": str(run_dir / "checkpoints"),
        "CHECKPOINT_DIR_SNO": str(run_dir / "checkpoints/pipeline_sno"),
        "CHECKPOINT_DIR_DNO": str(run_dir / "checkpoints/pipeline_dno"),
    }
    config_count = 0
    for cell in result["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "class Config:" in source:
            source, changed = replace_config(source, replacements)
            config_count += changed
        # Original installers fetch moving Transformers main or force an old
        # NumPy. Retain their text as comments, then use the release requirements.
        if "!pip install" in source or "CRITICAL: NumPy Version Lock" in source:
            original = "".join("# ORIGINAL INSTALLER: " + line + "\n" for line in source.splitlines())
            source = (original + "\nimport subprocess, sys\n" +
                      f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', {str(artifact_root / 'requirements' / requirements)!r}])\n")
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    if config_count < 2:
        raise ValueError("Expected dataset and output settings were not found")
    bootstrap = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": ["import os\n", f"os.chdir({str(artifact_root)!r})\n",
                   f"print('Release run: {name}; original evidence remains under codes/ and results/')\n"],
    }
    result["cells"].insert(0, bootstrap)
    for key in ["widgets", "papermill"]:
        result.get("metadata", {}).pop(key, None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", choices=NAMES)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    properties = ["cut_vertex", "connectivity", "bipartite", "planarity",
                  "outer_planarity", "hamiltonian", "k_colorability", "completeness"]
    missing = [str(dataset / p / c) for p in properties for c in ["yes", "s_no", "d_no"]
               if not (dataset / p / c).is_dir()]
    if missing:
        parser.error(f"Expected an extracted dataset with all property/category folders; missing {missing[0]}")
    if args.notebook == "mllm_model_finetune" and not (dataset / "reasoning_texts.json").is_file():
        parser.error("MLLM fine-tuning requires the randomized dataset with reasoning_texts.json")
    run_name = args.run_name or args.notebook
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in {".", ".."}:
        parser.error("--run-name must be a simple directory name")
    run_dir = ROOT / "runs" / run_name
    output = run_dir / "run.ipynb"
    if output.exists():
        parser.error(f"Run already exists: {output}; select a new --run-name")
    notebook = json.loads((ROOT / "codes" / (args.notebook + ".ipynb")).read_text(encoding="utf-8"))
    prepared = make_copy(notebook, args.notebook, dataset, run_dir, ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(prepared, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(output)
    print(f"Dataset: {dataset}")
    print(f"New outputs: {run_dir / 'outputs'}")
    if args.notebook == "mllm_model_evaluation_3_wr":
        print("NOTE: this sets the intended non-randomized input for a NEW run; historical input provenance is unresolved.")


if __name__ == "__main__":
    main()
