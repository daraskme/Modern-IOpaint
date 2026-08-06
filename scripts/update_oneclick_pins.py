#!/usr/bin/env python3
"""Download a selected uv Windows release asset and pin it in bootstrap.ps1."""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOTSTRAP = REPOSITORY_ROOT / "installer" / "oneclick" / "bootstrap.ps1"
UV_ASSET_TEMPLATE = (
    "https://github.com/astral-sh/uv/releases/download/"
    "{version}/uv-x86_64-pc-windows-msvc.zip"
)


def download_asset(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, destination.open(
        "wb"
    ) as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def replace_assignment(text: str, variable: str, value: str) -> str:
    pattern = rf'(?m)^\${re.escape(variable)}\s*=\s*"[^"]*"\s*$'
    replacement = f'${variable} = "{value}"'
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"could not find exactly one ${variable} assignment")
    return updated


def update_bootstrap(bootstrap_path: Path, version: str, url: str) -> None:
    with tempfile.TemporaryDirectory(prefix="modern-iopaint-uv-pin-") as temp_dir:
        archive_path = Path(temp_dir) / "uv-windows.zip"
        print(f"Downloading {url}", flush=True)
        sha256 = download_asset(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = [Path(name).name.lower() for name in archive.namelist()]
            if "uv.exe" not in members:
                raise ValueError("downloaded release asset does not contain uv.exe")

    text = bootstrap_path.read_text(encoding="utf-8")
    text = replace_assignment(text, "UvVersion", version)
    text = replace_assignment(text, "UvUrl", url)
    text = replace_assignment(text, "UvSha256", sha256)
    bootstrap_path.write_text(text, encoding="utf-8")
    print(f"Pinned uv {version} ({sha256}) in {bootstrap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="exact uv GitHub release tag chosen by the maintainer",
    )
    parser.add_argument(
        "--url",
        help="override the Windows x86-64 release ZIP URL",
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=DEFAULT_BOOTSTRAP,
        help=f"PowerShell file to update (default: {DEFAULT_BOOTSTRAP})",
    )
    args = parser.parse_args()
    url = args.url or UV_ASSET_TEMPLATE.format(version=args.version)
    update_bootstrap(args.bootstrap.resolve(), args.version, url)


if __name__ == "__main__":
    main()
