import glob
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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
    FLUX_FILL_NAME,
)
from modern_iopaint.schema import ModelCategory, ModelInfo, ModelType
from modern_iopaint.model_metadata import load_local_model_metadata
from modern_iopaint.model.original_sd_configs import load_original_config
from modern_iopaint.model_manifest import (
    DownloadSpec,
    ModelManifestRecord,
    integrated_bundle_model_names,
    load_model_manifest,
)


QUARANTINED_MODEL_NAME_PARTS = ("anytext", "brushnet", "powerpaint")
QWEN_PRECISION_ENV = "MODERN_IOPAINT_QWEN_PRECISION"
QWEN_RANK_ENV = "MODERN_IOPAINT_QWEN_RANK"
QWEN_LIGHTNING_STEPS_ENV = "MODERN_IOPAINT_QWEN_LIGHTNING_STEPS"
FLUX_PRECISION_ENV = "MODERN_IOPAINT_FLUX_PRECISION"
BFL_FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
BFL_FLUX_FILL_MODEL_URL = "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
_nunchaku_error_logged = False


class FluxAccessError(RuntimeError):
    """Actionable access failure for the gated FLUX.1-Fill base repository."""

    code = "flux_access_required"

    def __init__(self, cause: Exception, *, offline: bool = False):
        self.repo_id = BFL_FLUX_FILL_REPO
        self.model_url = BFL_FLUX_FILL_MODEL_URL
        self.offline = offline
        self.status_code = 403
        detail = (
            "Access could not be checked while HF_HUB_OFFLINE=1. "
            if offline
            else "Hugging Face denied access to the gated repository. "
        )
        super().__init__(
            "FLUX.1-Fill-dev access is required. "
            f"{detail}"
            "To continue: (a) create or log into a Hugging Face account; "
            f"(b) accept the FLUX.1-dev license on {self.model_url}; and "
            "(c) provide an authorized token with `huggingface-cli login` "
            "(or `hf auth login`) or the HF_TOKEN environment variable. "
            "Modern-IOPaint will not bypass repository gating. "
            f"Original error: {cause}"
        )


@dataclass(frozen=True)
class ManifestModelArtifacts:
    record: ModelManifestRecord
    precision: str
    rank: str
    lightning_steps: int
    transformer_filename: str
    transformer_path: Path
    base_path: Path
    optional_component_paths: Mapping[str, Path]


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
    return _resolve_nunchaku_precision(
        precision,
        env_name=QWEN_PRECISION_ENV,
        backend_name="Qwen",
        option_name="qwen-precision",
    )


def _resolve_nunchaku_precision(
    precision: Optional[str],
    *,
    env_name: str,
    backend_name: str,
    option_name: str,
) -> str:
    env_precision = os.getenv(env_name)
    if env_precision and (not precision or precision == "auto"):
        precision = env_precision
    precision = str(precision or "auto").lower()
    if precision in ("int4", "fp4"):
        return precision
    if precision != "auto":
        raise ValueError(
            f"{backend_name} precision must be one of: auto, int4, fp4"
        )
    try:
        from nunchaku.utils import get_precision

        detected = get_precision()
    except Exception as error:
        raise RuntimeError(
            f"{backend_name} precision auto-detection requires a working "
            "nunchaku==1.2.1 "
            "installation and CUDA GPU. Install Nunchaku separately or pass "
            f"--{option_name} int4/fp4."
        ) from error
    if detected not in ("int4", "fp4"):
        raise RuntimeError(f"Nunchaku returned unsupported precision {detected!r}")
    return detected


def resolve_flux_precision(precision: Optional[str]) -> str:
    return _resolve_nunchaku_precision(
        precision,
        env_name=FLUX_PRECISION_ENV,
        backend_name="FLUX",
        option_name="flux-precision",
    )


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

    try:
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
    except Exception as error:
        if spec.repo != BFL_FLUX_FILL_REPO:
            raise

        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        offline = os.getenv("HF_HUB_OFFLINE", "").strip().lower() in (
            "1",
            "on",
            "true",
            "yes",
        )
        if (
            isinstance(error, GatedRepoError)
            or (isinstance(error, HfHubHTTPError) and status_code in (401, 403))
            or status_code in (401, 403)
            or offline
        ):
            raise FluxAccessError(error, offline=offline) from error
        raise


def _select_manifest_options(
    record: ModelManifestRecord,
    *,
    precision: str,
    rank: str,
    lightning_steps: int,
) -> Tuple[str, str, int]:
    if record.name == FLUX_FILL_NAME:
        selected_precision = resolve_flux_precision(precision)
    else:
        selected_precision = resolve_qwen_precision(precision)
    selected_rank = normalize_qwen_rank(rank)
    selected_steps = normalize_qwen_lightning_steps(lightning_steps)

    # Shared P2 callers carry Qwen's r32/lightning-8 defaults. A manifest
    # backend with a single supported value selects that value without making
    # those callers know about backend-specific rank/lightning shapes.
    if selected_rank not in record.ranks and len(record.ranks) == 1:
        selected_rank = record.ranks[0]
    if selected_steps not in record.lightning_steps and len(record.lightning_steps) == 1:
        selected_steps = record.lightning_steps[0]

    # filename() supplies the manifest-specific validation errors.
    record.filename(selected_precision, selected_rank, selected_steps)
    return selected_precision, selected_rank, selected_steps


def _optional_component_path(root: Path, spec: DownloadSpec) -> Optional[Path]:
    exact_patterns = [
        pattern
        for pattern in spec.allow_patterns
        if not any(character in pattern for character in "*?[")
    ]
    if len(exact_patterns) == 1:
        candidate = root / exact_patterns[0]
        return candidate if candidate.is_file() else None
    return root


def _resolve_optional_components(
    record: ModelManifestRecord,
    hub_cache: Path,
    *,
    local_files_only: bool,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name, spec in record.optional_components.items():
        try:
            root = _snapshot_download(
                spec,
                hub_cache,
                allow_patterns=spec.allow_patterns,
                local_files_only=local_files_only,
            )
        except Exception:
            if local_files_only:
                continue
            raise
        component_path = _optional_component_path(root, spec)
        if component_path is not None:
            paths[name] = component_path
    return paths


def resolve_manifest_model_artifacts(
    model: str,
    *,
    precision: str = "auto",
    rank: str = "r32",
    lightning_steps: int = 8,
    cache_dir: Optional[Path] = None,
) -> ManifestModelArtifacts:
    record = load_model_manifest().get(model)
    selected_precision, selected_rank, selected_steps = _select_manifest_options(
        record,
        precision=precision,
        rank=rank,
        lightning_steps=lightning_steps,
    )
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
            f"Nunchaku transformer is not present in the local Hub cache: "
            f"{transformer_path}. Run `modern-iopaint download --model {model}` "
            "with the same precision option first."
        )
    return ManifestModelArtifacts(
        record=record,
        precision=selected_precision,
        rank=selected_rank,
        lightning_steps=selected_steps,
        transformer_filename=filename,
        transformer_path=transformer_path,
        base_path=base_root,
        optional_component_paths=_resolve_optional_components(
            record, hub_cache, local_files_only=True
        ),
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
    selected_precision, selected_rank, selected_steps = _select_manifest_options(
        record,
        precision=precision,
        rank=rank,
        lightning_steps=lightning_steps,
    )
    filename = record.filename(selected_precision, selected_rank, selected_steps)
    hub_cache = get_hf_cache_dir(cache_dir)

    if is_manifest_model_downloaded(
        model,
        precision=selected_precision,
        rank=selected_rank,
        lightning_steps=selected_steps,
        cache_dir=cache_dir,
    ):
        artifacts = resolve_manifest_model_artifacts(
            model,
            precision=selected_precision,
            rank=selected_rank,
            lightning_steps=selected_steps,
            cache_dir=cache_dir,
        )
        missing_optional = set(record.optional_components) - set(
            artifacts.optional_component_paths
        )
        if not missing_optional:
            logger.info(
                "Manifest model {} ({}/{}/{}) is already downloaded",
                model,
                selected_precision,
                selected_rank,
                selected_steps,
            )
            return artifacts
        logger.info(
            "Manifest model {} core components are cached; downloading missing "
            "optional components: {}",
            model,
            sorted(missing_optional),
        )

    _preflight_disk_space(record, hub_cache)
    def download_transformer() -> Path:
        logger.info(
            "Downloading {} transformer {} from {} at revision {}",
            model,
            filename,
            record.repo,
            record.revision,
        )
        return _snapshot_download(
            record,
            hub_cache,
            allow_patterns=record.transformer_allow_patterns(
                selected_precision, selected_rank, selected_steps
            ),
            local_files_only=False,
        )

    def download_base() -> Path:
        logger.info(
            "Downloading {} base components from {} at revision {} "
            "(transformer/* excluded)",
            model,
            record.base.repo,
            record.base.revision,
        )
        return _snapshot_download(
            record.base,
            hub_cache,
            allow_patterns=record.base.allow_patterns,
            local_files_only=False,
        )

    # Check gated access before spending time on the separate quantized
    # transformer snapshot.
    if record.base.gated:
        base_root = download_base()
        transformer_root = download_transformer()
    else:
        transformer_root = download_transformer()
        base_root = download_base()

    optional_component_paths = _resolve_optional_components(
        record, hub_cache, local_files_only=False
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
        optional_component_paths=optional_component_paths,
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
    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    elif model in integrated_bundle_model_names():
        download_manifest_model(
            model,
            precision=precision,
            rank=rank,
            lightning_steps=lightning_steps,
            cache_dir=cache_dir,
        )
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
def get_sd_model_type(
    model_abs_path: str | os.PathLike[str],
) -> Optional[ModelType]:
    checkpoint_path = os.fspath(model_abs_path)
    if "inpaint" in Path(checkpoint_path).name.lower():
        model_type = ModelType.DIFFUSERS_SD_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionInpaintPipeline

        try:
            StableDiffusionInpaintPipeline.from_single_file(
                checkpoint_path,
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
                logger.info(f"Ignore non sdxl file: {checkpoint_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {checkpoint_path}: {e}")
            return
    return model_type


@lru_cache()
def get_sdxl_model_type(
    model_abs_path: str | os.PathLike[str],
) -> Optional[ModelType]:
    checkpoint_path = os.fspath(model_abs_path)
    if "inpaint" in checkpoint_path:
        model_type = ModelType.DIFFUSERS_SDXL_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionXLInpaintPipeline

        try:
            model = StableDiffusionXLInpaintPipeline.from_single_file(
                checkpoint_path,
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
                logger.info(f"Ignore non sdxl file: {checkpoint_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {checkpoint_path}: {e}")
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

        try:
            metadata = load_local_model_metadata(it)
        except ValueError as error:
            logger.error("Ignoring local checkpoint {}: {}", it, error)
            continue

        model_type_cache[it.name] = model_type
        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                category=metadata.category,
                is_single_file_diffusers=True,
                prediction_type=metadata.prediction_type,
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

        try:
            metadata = load_local_model_metadata(it)
        except ValueError as error:
            logger.error("Ignoring local checkpoint {}: {}", it, error)
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
                category=metadata.category,
                is_single_file_diffusers=True,
                prediction_type=metadata.prediction_type,
            )
        )
    return res


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from modern_iopaint.model import models

    # logger.info(f"Scanning inpaint models in {model_dir}")

    manifest_records = load_model_manifest().models
    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            manifest_record = manifest_records.get(name)
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                    category=(
                        manifest_record.category if manifest_record else m.category
                    ),
                    license_name=(
                        manifest_record.license_name if manifest_record else None
                    ),
                    license_url=(
                        manifest_record.license_url if manifest_record else None
                    ),
                    gated=manifest_record.gated if manifest_record else False,
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
                category=ModelCategory.INPAINT_PHOTO,
            )
        )
    return available_models


def _nunchaku_is_available() -> bool:
    global _nunchaku_error_logged
    try:
        from nunchaku import NunchakuQwenImageTransformer2DModel  # noqa: F401
        from nunchaku import NunchakuFluxTransformer2dModel  # noqa: F401
        from nunchaku import NunchakuT5EncoderModel  # noqa: F401
        from nunchaku.utils import get_precision  # noqa: F401

        return True
    except Exception as error:
        if not _nunchaku_error_logged:
            logger.warning(
                "Nunchaku models are hidden because Nunchaku could not be imported: {}. "
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
    flux_precision: str = "auto",
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
    for name in integrated_bundle_model_names():
        record = load_model_manifest().get(name)
        if name == FLUX_FILL_NAME:
            try:
                model_precision = resolve_flux_precision(flux_precision)
            except Exception as error:
                logger.warning(
                    "FLUX Fill is hidden because its precision could not be "
                    "resolved: {}",
                    error,
                )
                continue
            model_rank = "r32"
            model_lightning_steps = 0
            default_steps = 28
            default_guidance_scale = 30.0
        else:
            model_precision = precision
            model_rank = rank
            model_lightning_steps = lightning_steps
            default_steps = lightning_steps or 50
            default_guidance_scale = 1.0
        if not is_manifest_model_downloaded(
            name,
            precision=model_precision,
            rank=model_rank,
            lightning_steps=model_lightning_steps,
            cache_dir=cache_dir,
        ):
            continue
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=ModelType.DIFFUSERS_OTHER,
                category=record.category,
                default_steps=default_steps,
                default_guidance_scale=default_guidance_scale,
                license_name=record.license_name,
                license_url=record.license_url,
                gated=record.base.gated,
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
                    category=ModelCategory.INPAINT_PHOTO,
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
    flux_precision: str = "auto",
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
            flux_precision=flux_precision,
        )
    )
    return [
        model
        for model in available_models
        if not is_quarantined_model_name(model.name)
    ]
