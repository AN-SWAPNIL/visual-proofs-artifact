"""Compare each archived manual-test input with the supplied dataset archives.

This verifies displayed examples, not every image used by an evaluation run.
Requires Pillow; does not load models or execute notebook code.
"""

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def pixels(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.size, hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / "datasets/manifest.json").read_text(encoding="utf-8"))
    report = []
    for stem in ["mllm_model_evaluation_2.5", "mllm_model_evaluation_2.5_wr",
                 "mllm_model_evaluation_3", "mllm_model_evaluation_3_wr"]:
        notebook = json.loads((ROOT / "codes" / (stem + ".ipynb")).read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if "TEST_IMAGE_PATH =" in "".join(c["source"]))
        constants = {}
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"TEST_PROPERTY", "TEST_TYPE", "IMAGE_NUM"}:
                    constants[name] = ast.literal_eval(node.value)
        prop, category, number = (constants[n] for n in ["TEST_PROPERTY", "TEST_TYPE", "IMAGE_NUM"])
        relative = f"{prop}/{category}/{prop}_{category}_{number}.png"
        output = next(o for o in cell["outputs"] if "image/png" in o.get("data", {}))
        actual = pixels(base64.b64decode(output["data"]["image/png"]))
        matches = []
        for condition in ["r", "wr"]:
            archive = ROOT / "datasets/archives" / catalog["conditions"][condition]["archive"]
            with zipfile.ZipFile(archive) as z:
                expected = pixels(z.read("graphtrails_dataset/" + relative))
            if actual == expected:
                matches.append(condition)
        entry = {"notebook": "codes/" + stem + ".ipynb", "image": relative,
                 "dimensions": list(actual[0]), "rgb_sha256": actual[1],
                 "matching_collections": matches,
                 "scope": "Single archived manual-test input only, not verification of full evaluation input"}
        report.append(entry)
        print(f"{stem}: {relative}, {actual[0]}, matches {matches}")
        intended = "wr" if stem.endswith("_wr") else "r"
        if matches != [intended]:
            raise AssertionError(f"Manual input does not uniquely match intended collection: {stem}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All four manual examples match intended collections. This does not verify every evaluated input.")


if __name__ == "__main__":
    main()
