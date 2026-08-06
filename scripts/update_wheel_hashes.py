#!/usr/bin/env python3
"""Download every configured Nunchaku wheel and update its SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "modern_iopaint" / "gpu_wheels.json"
ALLOWED_HOST = "github.com"
ALLOWED_PATH_PREFIX = "/nunchaku-ai/nunchaku/releases/download/"


def sha256_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"refusing non-GitHub release URL: {url}")
    if not parsed.path.startswith(ALLOWED_PATH_PREFIX):
        raise ValueError(f"refusing unexpected GitHub release path: {url}")

    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheels = data.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise ValueError(f"{manifest_path} does not contain a non-empty wheels list")

    for index, wheel in enumerate(wheels, start=1):
        url = str(wheel.get("url", ""))
        filename = str(wheel.get("filename", ""))
        if not url or not filename:
            raise ValueError(f"wheel entry {index} is missing url or filename")
        decoded_name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        if decoded_name != filename:
            raise ValueError(
                f"wheel entry {index} filename does not match its URL: "
                f"{filename!r} != {decoded_name!r}"
            )
        print(f"[{index}/{len(wheels)}] hashing {filename}", flush=True)
        wheel["sha256"] = sha256_url(url)

    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    print(f"Updated {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest to update (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()
    update_manifest(args.manifest.resolve())


if __name__ == "__main__":
    main()
