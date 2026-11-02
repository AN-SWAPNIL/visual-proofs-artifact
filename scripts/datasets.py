"""Verify and safely extract the published image collections using only stdlib."""

import argparse
import csv
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def sha256(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(name):
    if "\\" in name:
        raise ValueError(f"Backslash in archive path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def read_manifest(condition):
    catalog = json.loads((ROOT / "datasets/manifest.json").read_text(encoding="utf-8"))
    entry = catalog["conditions"][condition]
    with (ROOT / "datasets" / entry["file_manifest"]).open(newline="", encoding="utf-8") as stream:
        files = {row["path"]: row for row in csv.DictReader(stream)}
    return entry, files


def verify_archive(archive, entry, files):
    with archive.open("rb") as stream:
        archive_matches = sha256(stream) == entry["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        names = {}
        for info in z.infolist():
            path = safe_member(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symbolic link in archive: {info.filename}")
            if info.is_dir():
                continue
            parts = path.parts
            # Kaggle downloads may add a wrapper directory. File identities,
            # rather than ZIP compression or wrapper names, are authoritative.
            if "graphtrails_dataset" not in parts:
                raise ValueError(f"Missing graphtrails_dataset root: {info.filename}")
            root_index = parts.index("graphtrails_dataset")
            relative = PurePosixPath(*parts[root_index + 1:]).as_posix()
            if relative in names:
                raise ValueError(f"Duplicate dataset path: {relative}")
            names[relative] = info
        missing, extra = set(files) - names.keys(), names.keys() - set(files)
        if missing or extra:
            raise ValueError(f"Dataset membership mismatch: {len(missing)} missing, {len(extra)} extra")
        for relative, expected in files.items():
            info = names[relative]
            if info.file_size != int(expected["bytes"]):
                raise ValueError(f"Size mismatch: {relative}")
            with z.open(info) as stream:
                if sha256(stream) != expected["sha256"]:
                    raise ValueError(f"Content checksum mismatch: {relative}")
    return names, archive_matches


def extract_verified(archive, names, destination):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for relative, info in names.items():
            target = destination.joinpath(*PurePosixPath(relative).parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Path escapes dataset destination: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "verify", "extract"])
    parser.add_argument("--condition", choices=["r", "wr"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        for condition in ["r", "wr"]:
            entry, _ = read_manifest(condition)
            print(f"{condition}: {entry['name']}, {entry['image_count']} images")
            print(f"  {entry['kaggle_url']}")
            print(f"  local archive: datasets/archives/{entry['archive']}")
        return
    if not args.condition:
        parser.error("--condition is required for verify and extract")
    entry, files = read_manifest(args.condition)
    archive = args.archive or ROOT / "datasets/archives" / entry["archive"]
    names, archive_matches = verify_archive(archive, entry, files)
    print(f"Verified {len(names)} image/metadata files for {args.condition}.")
    print("Archive bytes match reference." if archive_matches else
          "ZIP bytes differ, but all uncompressed image/metadata contents match reference.")
    if args.command == "extract":
        destination = args.destination or ROOT / "datasets/extracted" / args.condition / "graphtrails_dataset"
        extract_verified(archive, names, destination)
        print(f"Extracted to {destination.resolve()}")


if __name__ == "__main__":
    main()
