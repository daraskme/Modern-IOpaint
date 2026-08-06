import glob
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

if os.name == "nt":
    # Hub snapshots can share blobs across repositories. Avoid privileged
    # symlink creation on Windows systems without Developer Mode.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from loguru import logger

from modern_iopaint.const import (
    DEFAULT_MODEL_DIR,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
)
from modern_iopaint.schema import ModelInfo, ModelType
from modern_iopaint.model.original_sd_configs import load_original_config
from modern_iopaint.model_manifest import (
    DownloadSpec,
    ModelManifestRecord,
    integrated_model_names,
    load_model_manifest,
)


QUARANTINED_MODEL_NAME_PARTS = ("anytext", "brushnet", "powerpaint")
QWEN_PRECISION_ENV = "MODERN_IOPAINT_QWEN_PRECISION"
QWEN_RANK_ENV = "MODERN_IOPAINT_QWEN_RANK"
QWEN_LIGHTNING_STEPS_ENV = "MODERN_IOPAINT_QWEN_LIGHTNING_STEPS"
_nunchaku_error_logged = False


@dataclass(frozen=True)
class ManifestModelArtifacts:
    record: ModelManifestRecord
    precision: str
    rank: str
    lightning_steps: int
    transformer_filename: str
    transformer_path: Path
    base_path: Path


def get_model_root(cache_dir: Optional[Path] = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().absolute()
    return Path(os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)).expanduser().absolute()


def get_hf_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Resolve the Hub cache while preserving the existing --model-dir layout."""

    if cache_dir is not None:
        return get_model_root(cache_dir) / "huggingface" / "hub"
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().absolute()
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().absolute() / "hub"
    if os.getenv("XDG_CACHE_HOME"):
        return (
            Path(os.environ["XDG_CACHE_HOME"]).expanduser().absolute()
            / "huggingface"
            / "hub"
        )

    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def resolve_qwen_precision(precision: Optional[str]) -> str:
    env_precision = os.getenv(QWEN_PRECISION_ENV)
    if env_precision and (not precision or precision == "auto"):
        precision = env_precision
    precision = str(precision or "auto").lower()
    if precision in ("int4", "fp4"):
        return precision
    if precision != "auto":
        raise ValueError("Qwen precision must be one of: auto, int4, fp4")
    try:
        from nunchaku.utils import get_precision

        detected = get_precision()
    except Exception as error:
        raise RuntimeError(
            "Qwen precision auto-detection requires a working nunchaku==1.2.1 "
            "installation and CUDA GPU. Install Nunchaku separately or pass "
            "--qwen-precision int4/fp4."
        ) from error
    if detected not in ("int4", "fp4"):
        raise RuntimeError(f"Nunchaku returned unsupported precision {detected!r}")
    return detected


def normalize_qwen_rank(rank: Optional[str]) -> str:
    env_rank = os.getenv(QWEN_RANK_ENV)
    if env_rank and (not rank or rank == "r32"):
        rank = env_rank
    rank = str(rank or "r32").lower()
    if rank not in ("r32", "r128"):
        raise ValueError("Qwen rank must be one of: r32, r128")
    return rank


def normalize_qwen_lightning_steps(lightning_steps: Optional[int]) -> int:
    env_steps = os.getenv(QWEN_LIGHTNING_STEPS_ENV)
    if env_steps and (lightning_steps is None or lightning_steps == 8):
        lightning_steps = int(env_steps)
    if lightning_steps is None:
        lightning_steps = 8
    lightning_steps = int(lightning_steps)
    if lightning_steps not in (0, 4, 8):
        raise ValueError("Qwen lightning steps must be one of: 0, 4, 8")
    return lightning_steps


def _snapshot_download(
    spec: DownloadSpec,
    cache_dir: Path,
    *,
    allow_patterns: Tuple[str, ...],
    local_files_only: bool,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec.repo,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            allow_patterns=list(allow_patterns),
            ignore_patterns=list(spec.ignore_patterns),
            local_files_only=local_files_only,
        )
    )


def resolve_manifest_model_artifacts(
    model: str,
    *,
    precision: str = "auto",
    rank: str = "r32",
    lightning_steps: int = 8,
    cache_dir: Optional[Path] = None,
) -> ManifestModelArtifacts:
    record = load_model_manifest().get(model)
    selected_precision = resolve_qwen_precision(precision)
    selected_rank = normalize_qwen_rank(rank)
    selected_steps = normalize_qwen_lightning_steps(lightning_steps)
    filename = record.filename(selected_precision, selected_rank, selected_steps)
    hub_cache = get_hf_cache_dir(cache_dir)

    transformer_root = _snapshot_download(
        record,
        hub_cache,
        allow_patterns=record.transformer_allow_patterns(
            selected_precision, selected_rank, selected_steps
        ),
        local_files_only=True,
    )
    base_root = _snapshot_download(
        record.base,
        hub_cache,
        allow_patterns=record.base.allow_patterns,
        local_files_only=True,
    )
    transformer_path = transformer_root / filename
    if not transformer_path.is_file():
        raise FileNotFoundError(
            f"Qwen transformer is not present in the local Hub cache: "
            f"{transformer_path}. Run `modern-iopaint download --model {model}` "
            "with the same Qwen precision/rank/lightning options first."
        )
    return ManifestModelArtifacts(
        record=record,
        precision=selected_precision,
        rank=selected_rank,
        lightning_steps=selected_steps,
        transformer_filename=filename,
        transformer_path=transformer_path,
        base_path=base_root,
    )


def is_manifest_model_downloaded(
    model: str,
    *,
    precision: str = "auto",
    rank: str = "r32",
    lightning_steps: int = 8,
    cache_dir: Optional[Path] = None,
) -> bool:
    try:
        resolve_manifest_model_artifacts(
            model,
            precision=precision,
            rank=rank,
            lightning_steps=lightning_steps,
            cache_dir=cache_dir,
        )
        return True
    except Exception:
        return False


def _preflight_disk_space(record: ModelManifestRecord, hub_cache: Path) -> None:
    hub_cache.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(hub_cache).free
    required_bytes = record.approx_download_size_bytes
    if free_bytes < required_bytes:
        required_gib = required_bytes / (1024**3)
        free_gib = free_bytes / (1024**3)
        raise OSError(
            f"Insufficient free disk space for {record.name}: approximately "
            f"{required_gib:.1f} GiB is required, but only {free_gib:.1f} GiB "
            f"is free at {hub_cache}. Free disk space or select another cache "
            "with --model-dir / XDG_CACHE_HOME / HF_HUB_CACHE, then retry."
        )
    logger.info(
        "Disk preflight passed for {}: {:.1f} GiB free, approximately {:.1f} GiB required",
        record.name,
        free_bytes / (1024**3),
        required_bytes / (1024**3),
    )


def download_manifest_model(
    model: str,
    *,
    precision: str = "auto",
    rank: str = "r32",
    lightning_steps: int = 8,
    cache_dir: Optional[Path] = None,
) -> ManifestModelArtifacts:
    record = load_model_manifest().get(model)
    selected_precision = resolve_qwen_precision(precision)
    selected_rank = normalize_qwen_rank(rank)
    selected_steps = normalize_qwen_lightning_steps(lightning_steps)
    filename = record.filename(selected_precision, selected_rank, selected_steps)
    hub_cache = get_hf_cache_dir(cache_dir)

    if is_manifest_model_downloaded(
        model,
        precision=selected_precision,
        rank=selected_rank,
        lightning_steps=selected_steps,
        cache_dir=cache_dir,
    ):
        logger.info(
            "Manifest model {} ({}/{}/{}) is already downloaded",
            model,
            selected_precision,
            selected_rank,
            selected_steps,
        )
        return resolve_manifest_model_artifacts(
            model,
            precision=selected_precision,
            rank=selected_rank,
            lightning_steps=selected_steps,
            cache_dir=cache_dir,
        )

    _preflight_disk_space(record, hub_cache)
    logger.info(
        "Downloading {} transformer {} from {} at revision {}",
        model,
        filename,
        record.repo,
        record.revision,
    )
    transformer_root = _snapshot_download(
        record,
        hub_cache,
        allow_patterns=record.transformer_allow_patterns(
            selected_precision, selected_rank, selected_steps
        ),
        local_files_only=False,
    )
    logger.info(
        "Downloading {} base components from {} at revision {} "
        "(transformer/* excluded)",
        model,
        record.base.repo,
        record.base.revision,
    )
    base_root = _snapshot_download(
        record.base,
        hub_cache,
        allow_patterns=record.base.allow_patterns,
        local_files_only=False,
    )
    transformer_path = transformer_root / filename
    if not transformer_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face download completed without expected file {filename}"
        )
    logger.info("Finished downloading {} to {}", model, hub_cache)
    return ManifestModelArtifacts(
        record=record,
        precision=selected_precision,
        rank=selected_rank,
        lightning_steps=selected_steps,
        transformer_filename=filename,
        transformer_path=transformer_path,
        base_path=base_root,
    )


def is_quarantined_model_name(name: str) -> bool:
    normalized_name = name.lower()
    return any(part in normalized_name for part in QUARANTINED_MODEL_NAME_PARTS)


def cli_download_model(
    model: str,
    *,
    precision: str = "auto",
    rank: str = "r32",
    lightning_steps: int = 8,
    cache_dir: Optional[Path] = None,
):
    from modern_iopaint.model import models

    if is_quarantined_model_name(model):
        raise ValueError(
            f"Model {model!r} is quarantined for the active Diffusers compatibility tuple"
        )
    if model in integrated_model_names():
        download_manifest_model(
            model,
            precision=precision,
            rank=rank,
            lightning_steps=lightning_steps,
            cache_dir=cache_dir,
        )
    elif model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=model,
            cache_dir=str(get_hf_cache_dir(cache_dir)),
        )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


@lru_cache(maxsize=512)
def get_sd_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in Path(model_abs_path).name.lower():
        model_type = ModelType.DIFFUSERS_SD_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionInpaintPipeline

        try:
            StableDiffusionInpaintPipeline.from_single_file(
                model_abs_path,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                num_in_channels=9,
                original_config=load_original_config("v1"),
            )
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        except ValueError as e:
            if "[320, 4, 3, 3]" in str(e):
                model_type = ModelType.DIFFUSERS_SD
            else:
                logger.info(f"Ignore non sdxl file: {model_abs_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {model_abs_path}: {e}")
            return
    return model_type


@lru_cache()
def get_sdxl_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in model_abs_path:
        model_type = ModelType.DIFFUSERS_SDXL_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionXLInpaintPipeline

        try:
            model = StableDiffusionXLInpaintPipeline.from_single_file(
                model_abs_path,
                num_in_channels=9,
                original_config=load_original_config("xl"),
            )
            if model.unet.config.in_channels == 9:
                # https://github.com/huggingface/diffusers/issues/6610
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                model_type = ModelType.DIFFUSERS_SDXL
        except ValueError as e:
            if "[320, 4, 3, 3]" in str(e):
                model_type = ModelType.DIFFUSERS_SDXL
            else:
                logger.info(f"Ignore non sdxl file: {model_abs_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {model_abs_path}: {e}")
            return
    return model_type


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    cache_file = stable_diffusion_dir / "modern_iopaint_cache.json"
    model_type_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                model_type_cache = json.load(f)
                assert isinstance(model_type_cache, dict)
        except:
            pass

    res = []
    for it in stable_diffusion_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sd_model_type(model_abs_path)
        if model_type is None:
            continue

        model_type_cache[it.name] = model_type
        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    if stable_diffusion_dir.exists():
        with open(cache_file, "w", encoding="utf-8") as fw:
            json.dump(model_type_cache, fw, indent=2, ensure_ascii=False)

    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    sdxl_cache_file = stable_diffusion_xl_dir / "modern_iopaint_cache.json"
    sdxl_model_type_cache = {}
    if sdxl_cache_file.exists():
        try:
            with open(sdxl_cache_file, "r", encoding="utf-8") as f:
                sdxl_model_type_cache = json.load(f)
                assert isinstance(sdxl_model_type_cache, dict)
        except:
            pass

    for it in stable_diffusion_xl_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = sdxl_model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sdxl_model_type(model_abs_path)
        if model_type is None:
            continue

        sdxl_model_type_cache[it.name] = model_type
        if stable_diffusion_xl_dir.exists():
            with open(sdxl_cache_file, "w", encoding="utf-8") as fw:
                json.dump(sdxl_model_type_cache, fw, indent=2, ensure_ascii=False)

        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    return res


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from modern_iopaint.model import models

    # logger.info(f"Scanning inpaint models in {model_dir}")

    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
    return res


def scan_diffusers_models(cache_dir: Optional[Path] = None) -> List[ModelInfo]:
    available_models = []
    hub_cache_dir = get_hf_cache_dir(cache_dir)
    # logger.info(f"Scanning diffusers models in {hub_cache_dir}")
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(hub_cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        try:
            with open(it, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        _class_name = data["_class_name"]
        name = folder_name_to_show_name(it.parent.parent.parent.name)
        if name in diffusers_model_names:
            continue
        if is_quarantined_model_name(name):
            continue
        if _class_name == DIFFUSERS_SD_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD
        elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL
        elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL_INPAINT
        elif _class_name in [
            "StableDiffusionInstructPix2PixPipeline",
            "PaintByExamplePipeline",
            "KandinskyV22InpaintPipeline",
        ]:
            model_type = ModelType.DIFFUSERS_OTHER
        else:
            continue

        diffusers_model_names.append(name)
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=model_type,
            )
        )
    return available_models


def _nunchaku_is_available() -> bool:
    global _nunchaku_error_logged
    try:
        from nunchaku import NunchakuQwenImageTransformer2DModel  # noqa: F401
        from nunchaku.utils import get_precision  # noqa: F401

        return True
    except Exception as error:
        if not _nunchaku_error_logged:
            logger.warning(
                "Qwen models are hidden because Nunchaku could not be imported: {}. "
                "Install nunchaku==1.2.1 separately in this environment; LaMa, "
                "SD, and SDXL remain available.",
                error,
            )
            _nunchaku_error_logged = True
        return False


def scan_manifest_models(
    cache_dir: Optional[Path] = None,
    *,
    qwen_precision: str = "auto",
    qwen_rank: str = "r32",
    qwen_lightning_steps: int = 8,
) -> List[ModelInfo]:
    if not _nunchaku_is_available():
        return []

    try:
        precision = resolve_qwen_precision(qwen_precision)
        rank = normalize_qwen_rank(qwen_rank)
        lightning_steps = normalize_qwen_lightning_steps(qwen_lightning_steps)
    except Exception as error:
        logger.warning(
            "Qwen models are hidden because their runtime options could not be "
            "resolved: {}. LaMa, SD, and SDXL remain available.",
            error,
        )
        return []

    available_models = []
    for name in integrated_model_names():
        if not is_manifest_model_downloaded(
            name,
            precision=precision,
            rank=rank,
            lightning_steps=lightning_steps,
            cache_dir=cache_dir,
        ):
            continue
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=ModelType.DIFFUSERS_OTHER,
                default_steps=lightning_steps or 50,
                default_guidance_scale=1.0,
            )
        )
    return available_models


def _scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        with open(it, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                logger.error(
                    f"Failed to load {it}, please try revert from original model or fix model_index.json by hand."
                )
                continue

            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.name)
            if name in diffusers_model_names:
                continue
            if is_quarantined_model_name(name):
                continue
            elif _class_name == DIFFUSERS_SD_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD
            elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
            elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL
            elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                continue

            diffusers_model_names.append(name)
            available_models.append(
                ModelInfo(
                    name=name,
                    path=str(it.parent.absolute()),
                    model_type=model_type,
                )
            )
    return available_models


def scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_dir))
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_xl_dir))
    return available_models


def scan_models(
    cache_dir: Optional[Path] = None,
    *,
    qwen_precision: str = "auto",
    qwen_rank: str = "r32",
    qwen_lightning_steps: int = 8,
) -> List[ModelInfo]:
    model_dir = get_model_root(cache_dir)
    available_models = []
    available_models.extend(scan_inpaint_models(model_dir))
    available_models.extend(scan_single_file_diffusion_models(model_dir))
    available_models.extend(scan_diffusers_models(cache_dir))
    available_models.extend(scan_converted_diffusers_models(model_dir))
    available_models.extend(
        scan_manifest_models(
            cache_dir,
            qwen_precision=qwen_precision,
            qwen_rank=qwen_rank,
            qwen_lightning_steps=qwen_lightning_steps,
        )
    )
    return [
        model
        for model in available_models
        if not is_quarantined_model_name(model.name)
    ]
