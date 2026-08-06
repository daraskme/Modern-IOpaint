from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
SUPPORTED_MANIFEST_VERSION = 1
QWEN_PRECISIONS = ("int4", "fp4")
QWEN_RANKS = ("r32", "r128")
QWEN_LIGHTNING_STEPS = (0, 4, 8)


class ManifestError(ValueError):
    """Raised when the bundled model manifest is malformed or unsupported."""


@dataclass(frozen=True)
class DownloadSpec:
    repo: str
    revision: str
    revision_note: str
    approx_download_size_bytes: int
    license_name: str
    license_url: str
    gated: bool
    allow_patterns: Tuple[str, ...]
    ignore_patterns: Tuple[str, ...]
    sha256: Optional[Mapping[str, str]]


@dataclass(frozen=True)
class ModelManifestRecord(DownloadSpec):
    name: str
    integrated: bool
    backend: str
    approx_download_size: str
    filename_templates: Mapping[str, str]
    precisions: Tuple[str, ...]
    ranks: Tuple[str, ...]
    lightning_steps: Tuple[int, ...]
    base: DownloadSpec

    def filename(self, precision: str, rank: str, lightning_steps: int) -> str:
        if precision not in self.precisions:
            raise ManifestError(
                f"Unsupported precision {precision!r} for {self.name}; "
                f"choose one of {list(self.precisions)}"
            )
        if rank not in self.ranks:
            raise ManifestError(
                f"Unsupported rank {rank!r} for {self.name}; "
                f"choose one of {list(self.ranks)}"
            )
        if lightning_steps not in self.lightning_steps:
            raise ManifestError(
                f"Unsupported lightning step count {lightning_steps!r} for "
                f"{self.name}; choose one of {list(self.lightning_steps)}"
            )
        template_key = "none" if lightning_steps == 0 else str(lightning_steps)
        try:
            template = self.filename_templates[template_key]
        except KeyError as error:
            raise ManifestError(
                f"Manifest record {self.name!r} has no filename template for "
                f"lightning_steps={lightning_steps}"
            ) from error
        return template.format(precision=precision, rank=rank)

    def transformer_allow_patterns(
        self, precision: str, rank: str, lightning_steps: int
    ) -> Tuple[str, ...]:
        filename = self.filename(precision, rank, lightning_steps)
        return tuple(pattern.format(filename=filename) for pattern in self.allow_patterns)


@dataclass(frozen=True)
class ModelManifest:
    version: int
    models: Mapping[str, ModelManifestRecord]

    def get(self, name: str, *, integrated_only: bool = True) -> ModelManifestRecord:
        try:
            record = self.models[name]
        except KeyError as error:
            raise ManifestError(
                f"Unknown manifest model {name!r}; choose one of "
                f"{sorted(self.models)}"
            ) from error
        if integrated_only and not record.integrated:
            raise ManifestError(
                f"Manifest model {name!r} is a placeholder and is not integrated"
            )
        return record


def _require(record: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in record:
        raise ManifestError(f"Missing {location}.{key}")
    return record[key]


def _parse_hashes(value: Any, location: str) -> Optional[Mapping[str, str]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError(f"{location}.sha256 must be null or an object")
    hashes: Dict[str, str] = {}
    for filename, digest in value.items():
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ManifestError(
                f"{location}.sha256 keys and values must both be strings"
            )
        normalized = digest.lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ManifestError(
                f"{location}.sha256[{filename!r}] is not a SHA-256 digest"
            )
        hashes[filename] = normalized
    return hashes


def _parse_download_spec(data: Mapping[str, Any], location: str) -> DownloadSpec:
    allow_patterns = _require(data, "allow_patterns", location)
    ignore_patterns = _require(data, "ignore_patterns", location)
    if not isinstance(allow_patterns, list) or not all(
        isinstance(item, str) for item in allow_patterns
    ):
        raise ManifestError(f"{location}.allow_patterns must be a list of strings")
    if not isinstance(ignore_patterns, list) or not all(
        isinstance(item, str) for item in ignore_patterns
    ):
        raise ManifestError(f"{location}.ignore_patterns must be a list of strings")

    size = _require(data, "approx_download_size_bytes", location)
    if not isinstance(size, int) or size <= 0:
        raise ManifestError(
            f"{location}.approx_download_size_bytes must be a positive integer"
        )
    gated = _require(data, "gated", location)
    if not isinstance(gated, bool):
        raise ManifestError(f"{location}.gated must be a boolean")

    spec = DownloadSpec(
        repo=str(_require(data, "repo", location)),
        revision=str(_require(data, "revision", location)),
        revision_note=str(_require(data, "revision_note", location)),
        approx_download_size_bytes=size,
        license_name=str(_require(data, "license_name", location)),
        license_url=str(_require(data, "license_url", location)),
        gated=gated,
        allow_patterns=tuple(allow_patterns),
        ignore_patterns=tuple(ignore_patterns),
        sha256=_parse_hashes(_require(data, "sha256", location), location),
    )
    if not spec.repo or not spec.revision or not spec.license_name or not spec.license_url:
        raise ManifestError(f"{location} contains an empty required string")
    return spec


def _parse_model(name: str, data: Mapping[str, Any]) -> ModelManifestRecord:
    location = f"models.{name}"
    common = _parse_download_spec(data, location)
    base_data = _require(data, "base", location)
    if not isinstance(base_data, dict):
        raise ManifestError(f"{location}.base must be an object")
    base = _parse_download_spec(base_data, f"{location}.base")
    if "transformer/*" not in base.ignore_patterns:
        raise ManifestError(
            f"{location}.base.ignore_patterns must exclude transformer/*"
        )

    templates = _require(data, "filename_templates", location)
    if not isinstance(templates, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in templates.items()
    ):
        raise ManifestError(f"{location}.filename_templates must be a string map")

    raw_precisions = _require(data, "precisions", location)
    raw_ranks = _require(data, "ranks", location)
    raw_lightning_steps = _require(data, "lightning_steps", location)
    if not isinstance(raw_precisions, list) or not isinstance(raw_ranks, list):
        raise ManifestError(f"{location}.precisions and ranks must be lists")
    if not isinstance(raw_lightning_steps, list):
        raise ManifestError(f"{location}.lightning_steps must be a list")
    precisions = tuple(raw_precisions)
    ranks = tuple(raw_ranks)
    lightning_steps = tuple(raw_lightning_steps)
    if not precisions or not set(precisions).issubset(QWEN_PRECISIONS):
        raise ManifestError(f"{location}.precisions contains unsupported values")
    if not ranks or not set(ranks).issubset(QWEN_RANKS):
        raise ManifestError(f"{location}.ranks contains unsupported values")
    if not lightning_steps or not set(lightning_steps).issubset(
        QWEN_LIGHTNING_STEPS
    ):
        raise ManifestError(f"{location}.lightning_steps contains unsupported values")

    integrated = _require(data, "integrated", location)
    if not isinstance(integrated, bool):
        raise ManifestError(f"{location}.integrated must be a boolean")

    record = ModelManifestRecord(
        **common.__dict__,
        name=name,
        integrated=integrated,
        backend=str(_require(data, "backend", location)),
        approx_download_size=str(_require(data, "approx_download_size", location)),
        filename_templates=dict(templates),
        precisions=precisions,
        ranks=ranks,
        lightning_steps=lightning_steps,
        base=base,
    )
    for precision in record.precisions:
        for rank in record.ranks:
            for steps in record.lightning_steps:
                filename = record.filename(precision, rank, steps)
                if not filename.endswith(".safetensors"):
                    raise ManifestError(
                        f"Resolved transformer filename is not safetensors: {filename}"
                    )
    return record


@lru_cache(maxsize=4)
def load_model_manifest(path: Optional[str] = None) -> ModelManifest:
    manifest_path = Path(path) if path else MANIFEST_PATH
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Unable to load model manifest {manifest_path}: {error}") from error

    if not isinstance(data, dict):
        raise ManifestError("Model manifest root must be an object")
    version = _require(data, "manifest_version", "manifest")
    if version != SUPPORTED_MANIFEST_VERSION:
        raise ManifestError(
            f"Unsupported model manifest version {version!r}; "
            f"expected {SUPPORTED_MANIFEST_VERSION}"
        )
    raw_models = _require(data, "models", "manifest")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ManifestError("manifest.models must be a non-empty object")

    models = {
        name: _parse_model(name, record)
        for name, record in raw_models.items()
        if isinstance(name, str) and isinstance(record, dict)
    }
    if len(models) != len(raw_models):
        raise ManifestError("Every manifest model must have a string name and object value")
    return ModelManifest(version=version, models=models)


def integrated_model_names() -> Tuple[str, ...]:
    manifest = load_model_manifest()
    return tuple(name for name, record in manifest.models.items() if record.integrated)
