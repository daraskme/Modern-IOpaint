"""Pin manifest revisions and populate available LFS SHA-256 metadata.

This maintenance script intentionally performs Hugging Face API requests. It is
not imported by Modern-IOPaint and must be run explicitly by a maintainer::

    python scripts/update_manifest_hashes.py

The Hub API exposes SHA-256 values for LFS-backed files. Small Git-backed files
do not have API SHA-256 metadata and are omitted from the resulting hash maps.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "modern_iopaint" / "model_manifest.json"


def matches(filename: str, allow: Iterable[str], ignore: Iterable[str]) -> bool:
    allowed = any(fnmatch.fnmatch(filename, pattern) for pattern in allow)
    ignored = any(fnmatch.fnmatch(filename, pattern) for pattern in ignore)
    return allowed and not ignored


def sibling_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        digest = lfs.get("sha256") or lfs.get("oid")
    else:
        digest = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if isinstance(digest, str) and digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if isinstance(digest, str) and len(digest) == 64:
        return digest.lower()
    return None


def pin_download_spec(api: HfApi, spec: dict[str, Any], allow: list[str]) -> None:
    info = api.model_info(
        repo_id=spec["repo"],
        revision=spec["revision"],
        files_metadata=True,
    )
    if not info.sha:
        raise RuntimeError(f"Hub API returned no commit SHA for {spec['repo']}")

    hashes: dict[str, str] = {}
    for sibling in info.siblings or []:
        filename = sibling.rfilename
        if not matches(filename, allow, spec["ignore_patterns"]):
            continue
        digest = sibling_sha256(sibling)
        if digest is not None:
            hashes[filename] = digest

    spec["revision"] = info.sha
    spec["revision_note"] = "Pinned by scripts/update_manifest_hashes.py."
    spec["sha256"] = dict(sorted(hashes.items())) or None


def transformer_filenames(record: dict[str, Any]) -> list[str]:
    filenames = []
    for precision in record["precisions"]:
        for rank in record["ranks"]:
            for steps in record["lightning_steps"]:
                key = "none" if steps == 0 else str(steps)
                filenames.append(
                    record["filename_templates"][key].format(
                        precision=precision,
                        rank=rank,
                    )
                )
    return sorted(set(filenames))


def update_manifest(path: Path, include_placeholders: bool) -> None:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    api = HfApi()
    for name, record in manifest["models"].items():
        if not record.get("integrated", False) and not include_placeholders:
            print(f"SKIP {name} (placeholder)")
            continue

        print(f"PIN  {name}: {record['repo']}")
        pin_download_spec(api, record, transformer_filenames(record))

        base = record["base"]
        print(f"PIN  {name} base: {base['repo']}")
        pin_download_spec(api, base, base["allow_patterns"])

    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest path to update",
    )
    parser.add_argument(
        "--include-placeholders",
        action="store_true",
        help="also pin records that are intentionally not integrated yet",
    )
    args = parser.parse_args()
    update_manifest(args.manifest.resolve(), args.include_placeholders)
    print(f"Updated {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
