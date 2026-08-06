from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from modern_iopaint.schema import ModelCategory


SUPPORTED_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")
LOCAL_CHECKPOINT_CATEGORIES = (
    ModelCategory.INPAINT_PHOTO,
    ModelCategory.INPAINT_ILLUSTRATION,
)
SIDECAR_FIELDS = {"prediction_type", "category"}


@dataclass(frozen=True)
class LocalModelMetadata:
    sidecar_path: Path
    sidecar_exists: bool
    prediction_type: Optional[str]
    category: ModelCategory


def checkpoint_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return ``checkpoint-name.json`` for a ckpt/safetensors checkpoint."""

    return Path(checkpoint_path).with_suffix(".json")


def load_local_model_metadata(checkpoint_path: str | Path) -> LocalModelMetadata:
    sidecar_path = checkpoint_sidecar_path(checkpoint_path)
    if not sidecar_path.is_file():
        return LocalModelMetadata(
            sidecar_path=sidecar_path,
            sidecar_exists=False,
            prediction_type=None,
            category=ModelCategory.INPAINT_PHOTO,
        )

    try:
        with sidecar_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read checkpoint sidecar {sidecar_path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint sidecar {sidecar_path} must contain a JSON object")
    unknown_fields = set(data) - SIDECAR_FIELDS
    if unknown_fields:
        raise ValueError(
            f"Checkpoint sidecar {sidecar_path} contains unsupported fields: "
            f"{sorted(unknown_fields)}"
        )

    prediction_type = data.get("prediction_type")
    if prediction_type is not None and prediction_type not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(
            f"Checkpoint sidecar {sidecar_path} prediction_type must be one of "
            f"{list(SUPPORTED_PREDICTION_TYPES)}"
        )

    raw_category = data.get("category", ModelCategory.INPAINT_PHOTO.value)
    try:
        category = ModelCategory(raw_category)
    except ValueError as error:
        raise ValueError(
            f"Checkpoint sidecar {sidecar_path} category must be one of "
            f"{[category.value for category in LOCAL_CHECKPOINT_CATEGORIES]}"
        ) from error
    if category not in LOCAL_CHECKPOINT_CATEGORIES:
        raise ValueError(
            f"Checkpoint sidecar {sidecar_path} category must be one of "
            f"{[category.value for category in LOCAL_CHECKPOINT_CATEGORIES]}"
        )

    return LocalModelMetadata(
        sidecar_path=sidecar_path,
        sidecar_exists=True,
        prediction_type=prediction_type,
        category=category,
    )


def apply_local_model_metadata(pipeline, checkpoint_path: str | Path) -> LocalModelMetadata:
    """Apply scheduler metadata after a single-file pipeline is constructed."""

    metadata = load_local_model_metadata(checkpoint_path)
    if metadata.prediction_type is not None:
        pipeline.scheduler.register_to_config(
            prediction_type=metadata.prediction_type
        )
        logger.info(
            "Applied prediction_type={} from checkpoint sidecar {}",
            metadata.prediction_type,
            metadata.sidecar_path,
        )
    return metadata
