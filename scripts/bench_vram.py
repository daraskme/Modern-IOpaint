#!/usr/bin/env python3
"""Offline VRAM screening harness for Modern-IOPaint.

The harness never downloads model files. It uses Modern-IOPaint's local model
registry, real ModelManager load/switch/unload lifecycle, and normal
InpaintRequest paths. The first inference in every cell is a warmup; only the
second inference contributes the steady-state wall time.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Model discovery and pipeline construction must remain offline. These are set
# before importing Modern-IOPaint, Diffusers, Transformers, or Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GIB = 1024**3
FULL_MODELS = (
    "lama",
    "anime-lama",
    "migan",
    "qwen-image",
    "qwen-image-edit",
    "flux.1-fill-dev",
)
QUICK_MODELS = ("qwen-image", "lama")
RESOLUTIONS = (512, 1024, 2048)
PROFILES = ("fast", "conservative")
NUNCHAKU_MODELS = {"qwen-image", "qwen-image-edit", "flux.1-fill-dev"}
DIFFUSION_MODELS = NUNCHAKU_MODELS


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    manager_name: str
    is_diffusion: bool
    available: bool
    skip_reason: Optional[str] = None


@dataclass
class CellResult:
    model: str
    resolution: int
    profile: str
    status: str
    load_seconds: Optional[float]
    steady_seconds: Optional[float]
    max_allocated: Optional[int]
    max_reserved: Optional[int]
    free_before: Optional[int]
    total_before: Optional[int]
    free_after: Optional[int]
    total_after: Optional[int]
    peak_rss: Optional[int]
    settings: str


class RSSPeakSampler:
    """Sample absolute process RSS during one benchmark cell using psutil."""

    def __init__(self, process: Any, interval_seconds: float = 0.02):
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            self.peak = max(self.peak, int(self.process.memory_info().rss))
        except Exception:
            pass

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--quick",
        dest="preset",
        action="store_const",
        const="quick",
        help="benchmark qwen-image and lama at 1024 under both profiles (default)",
    )
    preset.add_argument(
        "--full",
        dest="preset",
        action="store_const",
        const="full",
        help="benchmark the full model/resolution/profile matrix",
    )
    parser.set_defaults(preset="quick")
    parser.add_argument(
        "--cap-gib",
        type=float,
        help="set a PyTorch per-process CUDA allocation cap in GiB",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional local SDXL .ckpt/.safetensors file (full preset only)",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "int4", "fp4"),
        default="auto",
        help="Nunchaku precision; auto uses Nunchaku GPU detection",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Modern-IOPaint model/cache root; defaults to the application setting",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    args = parser.parse_args()
    if args.cap_gib is not None and args.cap_gib <= 0:
        parser.error("--cap-gib must be greater than zero")
    if args.checkpoint is not None and args.preset != "full":
        parser.error("--checkpoint is supported by the --full preset")
    if args.device_index < 0:
        parser.error("--device-index must be non-negative")
    return args


def concise_error(error: BaseException, limit: int = 240) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    text = f"{error.__class__.__name__}: {message}"
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def version_of(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    except Exception as error:
        return f"unavailable ({concise_error(error, 100)})"


def driver_version() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 5,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(command, **kwargs)
        versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode == 0 and versions:
            return versions[0]
    except Exception:
        pass
    return "unavailable"


def format_gib(value: Optional[int]) -> str:
    return "—" if value is None else f"{value / GIB:.2f} GiB"


def format_seconds(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def markdown_text(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def nunchaku_capability() -> tuple[bool, Optional[str]]:
    try:
        from nunchaku import (  # noqa: F401
            NunchakuFluxTransformer2dModel,
            NunchakuQwenImageTransformer2DModel,
            NunchakuT5EncoderModel,
        )
        from nunchaku.utils import get_precision  # noqa: F401

        return True, None
    except Exception as error:
        return False, concise_error(error)


def resolve_precision(requested: str) -> tuple[Optional[str], Optional[str]]:
    if requested in ("int4", "fp4"):
        return requested, None
    try:
        from modern_iopaint.download import resolve_qwen_precision

        return resolve_qwen_precision("auto"), None
    except Exception as error:
        return None, concise_error(error)


def prepare_checkpoint(checkpoint: Path, torch: Any) -> tuple[Any, Optional[str]]:
    from modern_iopaint.download import get_sdxl_model_type
    from modern_iopaint.model_metadata import load_local_model_metadata
    from modern_iopaint.schema import ModelInfo, ModelType

    path = checkpoint.expanduser().absolute()
    if not path.is_file():
        return None, f"local checkpoint does not exist: {path}"
    if path.suffix.lower() not in (".ckpt", ".safetensors"):
        return None, "--checkpoint must be a .ckpt or .safetensors file"

    try:
        metadata = load_local_model_metadata(path)
        model_type = get_sdxl_model_type(str(path))
        if model_type not in (
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ):
            return None, f"checkpoint was not detected as SDXL: {model_type}"
        manager_name = f"benchmark-local-sdxl:{path.name}"
        return (
            ModelInfo(
                name=manager_name,
                path=str(path),
                model_type=model_type,
                category=metadata.category,
                is_single_file_diffusers=True,
                prediction_type=metadata.prediction_type,
                default_steps=8,
                default_guidance_scale=5.0,
            ),
            None,
        )
    except Exception as error:
        return None, concise_error(error)
    finally:
        # get_sdxl_model_type may construct a CPU pipeline while determining
        # whether a generic single-file checkpoint has 4 or 9 UNet channels.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_manager_class(extra_models: list[Any]):
    from modern_iopaint.model_manager import ModelManager

    class BenchmarkModelManager(ModelManager):
        """Add explicitly supplied local checkpoints to normal model discovery."""

        def __init__(self, *args: Any, **kwargs: Any):
            self._benchmark_extra_models = tuple(extra_models)
            super().__init__(*args, **kwargs)

        def scan_models(self) -> list[Any]:
            scanned = super().scan_models()
            for model_info in self._benchmark_extra_models:
                self.available_models[model_info.name] = model_info
            scanned_names = {model_info.name for model_info in scanned}
            return [
                *scanned,
                *(
                    model_info
                    for model_info in self._benchmark_extra_models
                    if model_info.name not in scanned_names
                ),
            ]

    return BenchmarkModelManager


def make_inputs(size: int):
    import numpy as np

    image = np.empty((size, size, 3), dtype=np.uint8)
    horizontal = np.linspace(28, 224, size, dtype=np.uint8)
    vertical = np.linspace(216, 44, size, dtype=np.uint8)
    image[:, :, 0] = horizontal[None, :]
    image[:, :, 1] = vertical[:, None]
    image[:, :, 2] = 118

    # A central, moderately sized object gives HDStrategy.CROP a realistic
    # connected mask while leaving enough context around it.
    center = size // 2
    radius_x = 256 if size == 2048 else max(48, size // 6)
    radius_y = 192 if size == 2048 else max(40, size // 8)
    yy, xx = np.ogrid[:size, :size]
    ellipse = (
        ((xx - center) / float(radius_x)) ** 2
        + ((yy - center) / float(radius_y)) ** 2
        <= 1.0
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[ellipse] = 255

    # Put visible structure under the mask instead of benchmarking an empty
    # synthetic canvas.
    image[ellipse, 0] = 242
    image[ellipse, 1] = 184
    image[ellipse, 2] = 56
    return image, mask


def make_request(spec: ModelSpec, size: int):
    from modern_iopaint.schema import HDStrategy, InpaintRequest, SDSampler

    is_2048 = size == 2048
    guidance = 1.0 if spec.manager_name.startswith("qwen-image") else 7.5
    if spec.manager_name == "flux.1-fill-dev":
        guidance = 30.0
    elif spec.manager_name.startswith("benchmark-local-sdxl:"):
        guidance = 5.0
    strength = (
        0.999
        if spec.manager_name.startswith("benchmark-local-sdxl:")
        else 1.0
    )

    crop_size = 1024
    crop_origin = (size - crop_size) // 2 if is_2048 else 0
    prompt = (
        "replace the masked object with a small glazed blue ceramic tile, "
        "preserve the surrounding scene"
        if spec.manager_name == "qwen-image-edit"
        else "a small glazed blue ceramic tile with natural highlights"
    )
    return InpaintRequest(
        hd_strategy=HDStrategy.CROP if is_2048 else HDStrategy.ORIGINAL,
        hd_strategy_crop_trigger_size=1024,
        hd_strategy_crop_margin=256,
        hd_strategy_resize_limit=1024,
        prompt=prompt if spec.is_diffusion else "",
        negative_prompt="blurry, distorted",
        use_croper=is_2048 and spec.is_diffusion,
        croper_x=crop_origin,
        croper_y=crop_origin,
        croper_width=crop_size if is_2048 else size,
        croper_height=crop_size if is_2048 else size,
        sd_steps=8,
        sd_guidance_scale=guidance,
        sd_strength=strength,
        sd_sampler=SDSampler.euler,
        sd_seed=42,
        sd_mask_blur=0,
        sd_match_histograms=False,
        sd_keep_unmasked_area=True,
    )


def settings_for(
    spec: ModelSpec,
    size: int,
    profile: str,
    request: Any,
    manager: Any = None,
) -> str:
    strategy = getattr(request.hd_strategy, "value", request.hd_strategy)
    if spec.is_diffusion:
        parts = [
            f"steps={request.sd_steps}",
            f"guidance={request.sd_guidance_scale:g}",
            f"strength={request.sd_strength:g}",
        ]
    else:
        parts = ["steps=n/a (feed-forward)"]

    if size == 2048 and spec.is_diffusion:
        parts.append(f"HDStrategy={strategy} + use_croper=1024x1024")
    elif size == 2048:
        parts.append(f"HDStrategy={strategy}, margin={request.hd_strategy_crop_margin}")
    else:
        parts.append(f"HDStrategy={strategy}")

    if spec.manager_name == "migan" and size > 512:
        parts.append("MIGAN native 512px mask crop")
    if spec.manager_name in NUNCHAKU_MODELS and size == 512:
        parts.append("backend minimum padded size=1024px")

    backend = getattr(manager, "model", None) if manager is not None else None
    pipeline = getattr(backend, "model", None)
    scheduler = getattr(pipeline, "scheduler", None)
    if spec.is_diffusion:
        if scheduler is not None:
            parts.append(f"scheduler={scheduler.__class__.__name__}")
        elif spec.manager_name.startswith("benchmark-local-sdxl:"):
            parts.append(
                f"sampler={getattr(request.sd_sampler, 'value', request.sd_sampler)}"
            )
        else:
            parts.append("scheduler=backend default")
    if spec.manager_name.startswith("qwen-image"):
        precision = getattr(backend, "precision", "selected precision")
        rank = getattr(backend, "rank", "r32")
        lightning = getattr(backend, "lightning_steps", 8)
        parts.append(f"{precision}/{rank}/lightning-{lightning}")
        runtime = getattr(backend, "runtime", None)
        blocks = getattr(runtime, "num_blocks_on_gpu", None)
        if blocks is not None:
            parts.append(f"GPU blocks={blocks}")
    elif spec.manager_name == "flux.1-fill-dev":
        precision = getattr(backend, "precision", "selected precision")
        parts.append(f"{precision}/r32")
        parts.append(
            "int4 T5=yes" if getattr(backend, "uses_nunchaku_t5", False) else "int4 T5=no"
        )
    elif spec.manager_name.startswith("benchmark-local-sdxl:"):
        prediction_type = None
        scheduler_config = getattr(scheduler, "config", None)
        if scheduler_config is not None:
            prediction_type = getattr(scheduler_config, "prediction_type", None)
        parts.append(f"fp16; prediction_type={prediction_type or 'checkpoint default'}")
    else:
        parts.append(f"runtime profile={profile} (not used by backend)")
    return "; ".join(parts)


def cuda_mem_info(torch: Any, device_index: int) -> tuple[Optional[int], Optional[int]]:
    try:
        free, total = torch.cuda.mem_get_info(device_index)
        return int(free), int(total)
    except Exception:
        return None, None


def skipped_rows(
    spec: ModelSpec,
    resolutions: tuple[int, ...],
    profile: str,
    reason: str,
    load_seconds: Optional[float] = None,
    status_kind: str = "SKIP",
) -> list[CellResult]:
    status = f"{status_kind}: {reason}"
    return [
        CellResult(
            model=spec.display_name,
            resolution=size,
            profile=profile,
            status=status,
            load_seconds=load_seconds,
            steady_seconds=None,
            max_allocated=None,
            max_reserved=None,
            free_before=None,
            total_before=None,
            free_after=None,
            total_after=None,
            peak_rss=None,
            settings=(
                "not run; planned: "
                + settings_for(spec, size, profile, make_request(spec, size))
            ),
        )
        for size in resolutions
    ]


def benchmark_cell(
    torch: Any,
    process: Any,
    manager: Any,
    spec: ModelSpec,
    size: int,
    profile: str,
    load_seconds: float,
    device_index: int,
) -> CellResult:
    image, mask = make_inputs(size)
    request = make_request(spec, size)
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    free_before, total_before = cuda_mem_info(torch, device_index)
    sampler = RSSPeakSampler(process)
    sampler.start()
    status = "OK"
    steady_seconds: Optional[float] = None
    try:
        warmup_output = manager(image, mask, request)
        torch.cuda.synchronize(device_index)
        warmup_output = None

        started = time.perf_counter()
        steady_output = manager(image, mask, request)
        torch.cuda.synchronize(device_index)
        steady_seconds = time.perf_counter() - started
        steady_output = None
    except Exception as error:
        status = f"ERROR: {concise_error(error)}"
        try:
            torch.cuda.synchronize(device_index)
        except Exception:
            pass
    peak_rss = sampler.stop()
    free_after, total_after = cuda_mem_info(torch, device_index)
    try:
        max_allocated = int(torch.cuda.max_memory_allocated(device_index))
        max_reserved = int(torch.cuda.max_memory_reserved(device_index))
    except Exception:
        max_allocated = None
        max_reserved = None

    if status != "OK":
        # Preserve the loaded model but release allocator cache left by a failed
        # attempt so later cells can still produce useful screening results.
        torch.cuda.empty_cache()

    return CellResult(
        model=spec.display_name,
        resolution=size,
        profile=profile,
        status=status,
        load_seconds=load_seconds,
        steady_seconds=steady_seconds,
        max_allocated=max_allocated,
        max_reserved=max_reserved,
        free_before=free_before,
        total_before=total_before,
        free_after=free_after,
        total_after=total_after,
        peak_rss=peak_rss,
        settings=settings_for(spec, size, profile, request, manager),
    )


def build_report(
    *,
    results: list[CellResult],
    timestamp: str,
    gpu_name: str,
    driver: str,
    total_vram: Optional[int],
    total_ram: int,
    precision_text: str,
    cap_text: str,
    versions: dict[str, str],
    preset: str,
) -> str:
    lines = [
        f"# Modern-IOPaint VRAM benchmark — {timestamp}",
        "",
        "> **Screening warning:** This is a screening/estimation tool only, **NOT certification of real N-GiB hardware behavior**. A PyTorch process cap does not emulate Windows display-memory reservation, allocator fragmentation, or non-PyTorch VRAM consumers.",
        "",
        f"- Preset: `{preset}`",
        f"- GPU: `{gpu_name}`",
        f"- NVIDIA driver: `{driver}`",
        f"- Total VRAM: `{format_gib(total_vram)}`",
        f"- Total system RAM: `{format_gib(total_ram)}`",
        f"- Nunchaku precision mode: `{precision_text}`",
        f"- Per-process VRAM cap: `{cap_text}`",
        "- Dependency tuple: "
        f"`python {versions['python']}`, `torch {versions['torch']}`, "
        f"`diffusers {versions['diffusers']}`, "
        f"`transformers {versions['transformers']}`, "
        f"`nunchaku {versions['nunchaku']}`",
        "",
        "Load time is measured separately from inference and repeated across that model/profile's resolution rows. The first generation in each cell is a warmup; steady-state wall time is the second generation. The first model in a profile uses `ModelManager` initialization, subsequent models use `ModelManager.switch()`, and profile boundaries use `ModelManager.unload()`.",
        "",
        "| Model | Resolution | Profile | Status / skip reason | Load time (s) | Steady-state wall (s) | Max allocated | Max reserved | Driver free/total before | Driver free/total after | Peak RSS | Steps/settings |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for result in results:
        before = (
            "—"
            if result.free_before is None or result.total_before is None
            else f"{format_gib(result.free_before)} / {format_gib(result.total_before)}"
        )
        after = (
            "—"
            if result.free_after is None or result.total_after is None
            else f"{format_gib(result.free_after)} / {format_gib(result.total_after)}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    markdown_text(result.model),
                    f"{result.resolution}×{result.resolution}",
                    result.profile,
                    markdown_text(result.status),
                    format_seconds(result.load_seconds),
                    format_seconds(result.steady_seconds),
                    format_gib(result.max_allocated),
                    format_gib(result.max_reserved),
                    before,
                    after,
                    format_gib(result.peak_rss),
                    markdown_text(result.settings),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(results: list[CellResult], report_path: Path) -> None:
    print(f"VRAM benchmark report: {report_path}")
    for result in results:
        if result.status == "OK":
            before = format_gib(result.free_before)
            after = format_gib(result.free_after)
            print(
                f"OK   {result.model:<24} {result.resolution:>4}px "
                f"{result.profile:<12} load={format_seconds(result.load_seconds)}s "
                f"steady={format_seconds(result.steady_seconds)}s "
                f"alloc={format_gib(result.max_allocated)} "
                f"reserved={format_gib(result.max_reserved)} "
                f"driver-free={before}->{after} rss={format_gib(result.peak_rss)} "
                f"[{result.settings}]"
            )
        else:
            status_kind, _, reason = result.status.partition(":")
            print(
                f"{status_kind:<5} {result.model:<24} {result.resolution:>4}px "
                f"{result.profile:<12} load={format_seconds(result.load_seconds)}s: "
                f"{reason.strip() or result.status}"
            )


def main() -> int:
    args = parse_args()
    try:
        import psutil
        import torch
    except Exception as error:
        print(f"Unable to import benchmark dependencies: {concise_error(error)}", file=sys.stderr)
        return 2

    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "diffusers": version_of("diffusers"),
        "transformers": version_of("transformers"),
        "nunchaku": version_of("nunchaku"),
    }
    total_ram = int(psutil.virtual_memory().total)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    results_dir = REPO_ROOT / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"results-{timestamp}.md"

    model_names = QUICK_MODELS if args.preset == "quick" else FULL_MODELS
    resolutions = (1024,) if args.preset == "quick" else RESOLUTIONS
    results: list[CellResult] = []
    gpu_name = "CUDA unavailable"
    total_vram: Optional[int] = None
    cap_text = "none"
    precision_text = args.precision

    if not torch.cuda.is_available():
        if args.cap_gib is not None:
            cap_text = f"{args.cap_gib:g} GiB requested; not applied (CUDA unavailable)"
        for profile in PROFILES:
            for name in model_names:
                spec = ModelSpec(name, name, name in DIFFUSION_MODELS, False)
                results.extend(
                    skipped_rows(spec, resolutions, profile, "CUDA is unavailable")
                )
            if args.checkpoint is not None:
                checkpoint_name = args.checkpoint.expanduser().name
                checkpoint_spec = ModelSpec(
                    display_name=f"sdxl:{checkpoint_name}",
                    manager_name=f"benchmark-local-sdxl:{checkpoint_name}",
                    is_diffusion=True,
                    available=False,
                )
                results.extend(
                    skipped_rows(
                        checkpoint_spec,
                        resolutions,
                        profile,
                        "CUDA is unavailable",
                    )
                )
        report = build_report(
            results=results,
            timestamp=timestamp,
            gpu_name=gpu_name,
            driver="unavailable",
            total_vram=total_vram,
            total_ram=total_ram,
            precision_text=precision_text,
            cap_text=cap_text,
            versions=versions,
            preset=args.preset,
        )
        report_path.write_text(report, encoding="utf-8")
        print_summary(results, report_path)
        return 0

    try:
        torch.cuda.set_device(args.device_index)
        properties = torch.cuda.get_device_properties(args.device_index)
    except Exception as error:
        print(f"Unable to select CUDA device {args.device_index}: {concise_error(error)}", file=sys.stderr)
        return 2

    # Existing SD/SDXL dtype selection checks for the exact string "cuda".
    # set_device above selects the requested index while this spelling keeps
    # that established fp16 path intact.
    device = torch.device("cuda")
    gpu_name = properties.name
    total_vram = int(properties.total_memory)
    if args.cap_gib is not None:
        fraction = args.cap_gib * GIB / total_vram
        if fraction > 1.0:
            print(
                f"--cap-gib {args.cap_gib:g} exceeds device VRAM "
                f"({total_vram / GIB:.2f} GiB)",
                file=sys.stderr,
            )
            return 2
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, args.device_index)
        except Exception as error:
            print(f"Unable to apply --cap-gib: {concise_error(error)}", file=sys.stderr)
            return 2
        cap_text = f"{args.cap_gib:g} GiB (fraction={fraction:.6f})"

    nunchaku_ok, nunchaku_reason = nunchaku_capability()
    selected_precision, precision_reason = resolve_precision(args.precision)
    if selected_precision is not None:
        precision_text = selected_precision
    else:
        precision_text = f"unresolved ({args.precision} requested: {precision_reason})"
    manager_precision = selected_precision or args.precision

    model_dir = args.model_dir.expanduser().absolute() if args.model_dir else None
    registry_error: Optional[str] = None
    try:
        from modern_iopaint.download import scan_models

        registered = scan_models(
            model_dir,
            qwen_precision=manager_precision,
            qwen_rank="r32",
            qwen_lightning_steps=8,
            flux_precision=manager_precision,
        )
        registered_names = {model_info.name for model_info in registered}
    except Exception as error:
        registered_names = set()
        registry_error = concise_error(error)

    checkpoint_info = None
    checkpoint_reason = None
    if args.checkpoint is not None:
        checkpoint_info, checkpoint_reason = prepare_checkpoint(args.checkpoint, torch)

    specs: list[ModelSpec] = []
    for name in model_names:
        reason = None
        if registry_error is not None:
            reason = f"local model registry failed: {registry_error}"
        elif name in NUNCHAKU_MODELS and not nunchaku_ok:
            reason = f"Nunchaku unavailable: {nunchaku_reason}"
        elif name in NUNCHAKU_MODELS and selected_precision is None:
            reason = f"Nunchaku precision unavailable: {precision_reason}"
        elif name not in registered_names:
            variant = (
                f" ({manager_precision}/r32/lightning-8)"
                if name.startswith("qwen-image")
                else f" ({manager_precision}/r32)" if name == "flux.1-fill-dev" else ""
            )
            reason = f"weights are not cached locally{variant}"
        specs.append(
            ModelSpec(
                display_name=name,
                manager_name=name,
                is_diffusion=name in DIFFUSION_MODELS,
                available=reason is None,
                skip_reason=reason,
            )
        )

    extra_models: list[Any] = []
    if args.checkpoint is not None:
        display_name = f"sdxl:{args.checkpoint.expanduser().name}"
        manager_name = (
            checkpoint_info.name
            if checkpoint_info is not None
            else f"benchmark-local-sdxl:{args.checkpoint.expanduser().name}"
        )
        specs.append(
            ModelSpec(
                display_name=display_name,
                manager_name=manager_name,
                is_diffusion=True,
                available=checkpoint_info is not None,
                skip_reason=checkpoint_reason,
            )
        )
        if checkpoint_info is not None:
            extra_models.append(checkpoint_info)

    BenchmarkModelManager = make_manager_class(extra_models)
    process = psutil.Process(os.getpid())
    manager_kwargs = {
        "model_cache_dir": model_dir,
        "no_half": False,
        "low_mem": False,
        "cpu_offload": False,
        "disable_nsfw": True,
        "sd_cpu_textencoder": False,
        "local_files_only": True,
        "enable_controlnet": False,
        "qwen_precision": manager_precision,
        "qwen_rank": "r32",
        "qwen_lightning_steps": 8,
        "flux_precision": manager_precision,
    }

    for profile in PROFILES:
        manager = None
        try:
            for spec in specs:
                if not spec.available:
                    results.extend(
                        skipped_rows(
                            spec,
                            resolutions,
                            profile,
                            spec.skip_reason or "model is unavailable",
                        )
                    )
                    continue

                load_started = time.perf_counter()
                try:
                    if manager is None:
                        manager = BenchmarkModelManager(
                            name=spec.manager_name,
                            device=device,
                            runtime_profile=profile,
                            **manager_kwargs,
                        )
                    else:
                        manager.switch(spec.manager_name)
                    torch.cuda.synchronize(args.device_index)
                    load_seconds = time.perf_counter() - load_started
                except Exception as error:
                    load_seconds = time.perf_counter() - load_started
                    results.extend(
                        skipped_rows(
                            spec,
                            resolutions,
                            profile,
                            f"model load failed: {concise_error(error)}",
                            load_seconds=load_seconds,
                            status_kind="ERROR",
                        )
                    )
                    if manager is not None and getattr(manager, "model", None) is None:
                        try:
                            manager.unload()
                        except Exception:
                            pass
                        manager = None
                    continue

                for size in resolutions:
                    results.append(
                        benchmark_cell(
                            torch=torch,
                            process=process,
                            manager=manager,
                            spec=spec,
                            size=size,
                            profile=profile,
                            load_seconds=load_seconds,
                            device_index=args.device_index,
                        )
                    )
        finally:
            if manager is not None:
                manager.unload()

    report = build_report(
        results=results,
        timestamp=timestamp,
        gpu_name=gpu_name,
        driver=driver_version(),
        total_vram=total_vram,
        total_ram=total_ram,
        precision_text=precision_text,
        cap_text=cap_text,
        versions=versions,
        preset=args.preset,
    )
    report_path.write_text(report, encoding="utf-8")
    print_summary(results, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
