from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from modern_iopaint.download import get_model_root


SETTINGS_FILENAME = "modern_iopaint_settings.json"


def license_settings_path(model_dir: Optional[Path] = None) -> Path:
    return get_model_root(model_dir) / SETTINGS_FILENAME


def _read_settings(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = license_settings_path(model_dir)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Unable to read license settings {}: {}", path, error)
        return {}
    return value if isinstance(value, dict) else {}


def is_license_accepted(model: str, model_dir: Optional[Path] = None) -> bool:
    settings = _read_settings(model_dir)
    acceptances = settings.get("license_acceptances", {})
    if not isinstance(acceptances, dict):
        return False
    record = acceptances.get(model, {})
    return isinstance(record, dict) and record.get("accepted") is True


def set_license_accepted(
    model: str,
    *,
    accepted: bool,
    license_name: str,
    license_url: str,
    model_dir: Optional[Path] = None,
) -> None:
    path = license_settings_path(model_dir)
    settings = _read_settings(model_dir)
    acceptances = settings.setdefault("license_acceptances", {})
    if not isinstance(acceptances, dict):
        acceptances = {}
        settings["license_acceptances"] = acceptances
    acceptances[model] = {
        "accepted": accepted,
        "accepted_at": datetime.now(timezone.utc).isoformat() if accepted else None,
        "license_name": license_name,
        "license_url": license_url,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)
