"""Recompute archived metrics from predictions; no inference or training."""

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, label):
    if not math.isclose(actual, expected, abs_tol=1e-10):
        raise AssertionError(f"{label}: recomputed={actual}, archived={expected}")


def metrics(rows):
    valid = [r for r in rows if int(r["pred_label"]) >= 0]
    if not valid:
        raise ValueError("No valid prediction rows")
    tp = sum(int(r["true_label"]) == 1 and int(r["pred_label"]) == 1 for r in valid)
    fp = sum(int(r["true_label"]) == 0 and int(r["pred_label"]) == 1 for r in valid)
    fn = sum(int(r["true_label"]) == 1 and int(r["pred_label"]) == 0 for r in valid)
    correct = sum(int(r["true_label"]) == int(r["pred_label"]) for r in valid)
    return {
        "accuracy": correct / len(valid),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "n_samples": len(valid),
    }


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["files"]:
        with (ROOT / entry["path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise AssertionError(f"Archived file changed: {entry['path']}")
    print(f"Provenance checksums: {len(provenance['files'])} files OK")
    for folder in sorted((ROOT / "results").iterdir()):
        if folder.name.startswith("vision"):
            continue
        finetuned = "fintetune" in folder.name
        summary_name = "mllm_finetune_results.json" if finetuned else "mllm_results.json"
        prefix = "mllm_finetune" if finetuned else "mllm"
        summary = json.loads((folder / summary_name).read_text(encoding="utf-8"))
        accuracies = []
        for experiment in ["exp1_s_no", "exp2_d_no"]:
            with (folder / f"{prefix}_{experiment}.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            actual = metrics(rows)
            for key, expected in summary[experiment]["metrics"].items():
                close(actual[key], expected, f"{folder.name}/{experiment}/{key}")
            for prop, expected in summary[experiment]["per_property"].items():
                prop_metrics = metrics([r for r in rows if r["property"] == prop])
                for key in ["accuracy", "f1"]:
                    close(prop_metrics[key], expected[key], f"{folder.name}/{experiment}/{prop}/{key}")
            invalid = sum(int(r["pred_label"]) < 0 for r in rows)
            print(f"{folder.name}/{experiment}: {len(rows)} rows, {invalid} errors, {actual['accuracy'] * 100:.4f}% accuracy")
            accuracies.append(actual["accuracy"])
        close(accuracies[1] - accuracies[0], summary["visual_proof_gain"], folder.name + "/VPG")

    vision = json.loads((ROOT / "results/vision_fintetune_results/final_results.json").read_text(encoding="utf-8"))
    checked = 0
    for experiment, data in vision["experiments"].items():
        for model, values in data["models"].items():
            matrices = [v["confusion_matrix"] for v in values["test_results"].values()]
            if not matrices:
                print(f"vision/{experiment}/{model}: summary retained; no per-property confusion matrices available to independently verify")
                continue
            tn, fp, fn, tp = [sum(m[i][j] for m in matrices) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
            total = tn + fp + fn + tp
            close((tn + tp) / total, values["overall"]["accuracy"], f"vision/{experiment}/{model}/accuracy")
            close(total, data["test_samples"], f"vision/{experiment}/{model}/sample count")
            checked += 1
    print(f"Vision overall accuracy: {checked} backbone summaries agree with property confusion matrices")
    print("Available detailed evidence matches its summaries. Read docs/REPRODUCIBILITY.md for design and provenance limits.")


if __name__ == "__main__":
    main()
