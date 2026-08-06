#!/usr/bin/env python3
"""Assemble the distributable Windows one-click ZIP."""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "installer" / "oneclick"
REQUIRED_FILES = ("run.bat", "bootstrap.ps1", "README.txt")
UV_PIN_NAMES = ("UvVersion", "UvUrl", "UvSha256")
UV_PIN_ASSIGNMENT_RE = re.compile(
    r'^\s*\$(?P<name>UvVersion|UvUrl|UvSha256)\s*=\s*'
    r'"(?P<value>[^"\r\n]*)"\s*$',
    re.MULTILINE,
)


def project_version() -> str:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return str(tomllib.load(pyproject_file)["project"]["version"])


def build_zip() -> Path:
    missing = [name for name in REQUIRED_FILES if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"one-click source files are missing: {', '.join(missing)}")

    bootstrap_text = (SOURCE_DIR / "bootstrap.ps1").read_text(encoding="utf-8")
    uv_pins = {
        match.group("name"): match.group("value")
        for match in UV_PIN_ASSIGNMENT_RE.finditer(bootstrap_text)
    }
    missing_pins = [name for name in UV_PIN_NAMES if name not in uv_pins]
    if missing_pins:
        raise ValueError(
            "bootstrap.ps1 is missing valid uv pin assignments for: "
            + ", ".join(f"${name}" for name in missing_pins)
        )

    placeholder_pins = [
        name for name in UV_PIN_NAMES if "PLACEHOLDER" in uv_pins[name]
    ]
    if placeholder_pins:
        raise ValueError(
            "bootstrap.ps1 still contains uv pin placeholders in "
            + ", ".join(f"${name}" for name in placeholder_pins)
            + "; run "
            "scripts/update_oneclick_pins.py before building the release ZIP"
        )

    output_dir = REPOSITORY_ROOT / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"modern-iopaint-oneclick-{project_version()}.zip"
    archive_root = "modern-iopaint-oneclick"
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in REQUIRED_FILES:
            archive.write(SOURCE_DIR / name, f"{archive_root}/{name}")
    return output_path


if __name__ == "__main__":
    print(build_zip())
