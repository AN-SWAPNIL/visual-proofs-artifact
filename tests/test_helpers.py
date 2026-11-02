import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


datasets = module("datasets")
prepare = module("prepare_notebook")
results = module("check_results")


class DatasetTests(unittest.TestCase):
    def test_reject_unsafe_paths(self):
        for path in ["../x", "/x", "C:/x", "root/../../x", "root\\x"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                datasets.safe_member(path)

    def fixture(self, directory, bad=False, wrapper=False):
        path = directory / "data.zip"
        payload = b"dataset image bytes"
        with zipfile.ZipFile(path, "w") as z:
            prefix = "wrapper/" if wrapper else ""
            z.writestr(prefix + "graphtrails_dataset/p/yes/a.png", payload if not bad else b"corrupted")
        files = {"p/yes/a.png": {"bytes": str(len(payload)), "sha256": hashlib.sha256(payload).hexdigest()}}
        entry = {"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return path, entry, files

    def test_verify_extract_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, entry, files = self.fixture(folder)
            names, matched = datasets.verify_archive(path, entry, files)
            self.assertTrue(matched)
            datasets.extract_verified(path, names, folder / "extracted")
            self.assertEqual((folder / "extracted/p/yes/a.png").read_bytes(), b"dataset image bytes")
            with self.assertRaises(FileExistsError):
                datasets.extract_verified(path, names, folder / "extracted")

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, entry, files = self.fixture(Path(tmp), bad=True)
            with self.assertRaises(ValueError):
                datasets.verify_archive(path, entry, files)

    def test_wrapper_and_repacked_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, entry, files = self.fixture(Path(tmp), wrapper=True)
            entry["archive_sha256"] = "not the same compressed bytes"
            _, matched = datasets.verify_archive(path, entry, files)
            self.assertFalse(matched)

    def test_traversal_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("graphtrails_dataset/../../outside", b"x")
            with self.assertRaises(ValueError):
                datasets.verify_archive(path, {"archive_sha256": ""}, {})


class NotebookTests(unittest.TestCase):
    def test_replaces_only_paths_and_preserves_comments(self):
        source = "class Config:\n    DATASET_ROOT = 'old'  # keep this comment\n    SEED = 42\n    RESULTS_DIR = 'old results'\n"
        changed, count = prepare.replace_config(source, {"DATASET_ROOT": "new", "RESULTS_DIR": "results"})
        self.assertEqual(count, 2)
        self.assertIn("# keep this comment", changed)
        self.assertIn("SEED = 42", changed)

    def test_original_unchanged_and_run_outputs_cleared(self):
        notebook = {"cells": [
            {"cell_type": "code", "source": ["!pip install something\n"], "metadata": {}, "outputs": [{"text": "old"}], "execution_count": 1},
            {"cell_type": "code", "source": ["class Config:\n    DATASET_ROOT = 'old'\n    RESULTS_DIR = 'old'\n    MAX_SAMPLES_PER_CLASS = 200\n"], "metadata": {}, "outputs": [], "execution_count": 2},
        ], "metadata": {"widgets": {"stale": 1}}}
        result = prepare.make_copy(notebook, "mllm_model_evaluation_3", Path("dataset"), Path("run"), ROOT)
        self.assertEqual(notebook["cells"][0]["execution_count"], 1)
        self.assertEqual(result["cells"][1]["outputs"], [])
        self.assertIn("ORIGINAL INSTALLER", "".join(result["cells"][1]["source"]))
        self.assertIn("MAX_SAMPLES_PER_CLASS = 200", "".join(result["cells"][2]["source"]))
        self.assertNotIn("widgets", result["metadata"])


class MetricTests(unittest.TestCase):
    def test_positive_class_is_yes(self):
        rows = [{"true_label": a, "pred_label": b} for a, b in [(1, 1), (1, 0), (0, 0), (0, 0)]]
        m = results.metrics(rows)
        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(m["precision"], 1)
        self.assertEqual(m["recall"], 0.5)

    def test_error_is_not_a_negative_prediction(self):
        rows = [{"true_label": 1, "pred_label": 1}, {"true_label": 0, "pred_label": -1}]
        self.assertEqual(results.metrics(rows)["n_samples"], 1)
        with self.assertRaises(ValueError):
            results.metrics([])


if __name__ == "__main__":
    unittest.main()
