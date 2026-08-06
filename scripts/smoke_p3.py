"""Phase P3 FLUX.1-Fill smoke test.

This script never downloads model weights. CUDA, Nunchaku, cached weights, and
the optional int4 T5 are checked per stage; unavailable prerequisites produce a
clear SKIP result instead of failing the complete smoke run.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path


GIB = 1024**3
MODEL_NAME = "flux.1-fill-dev"


class SkipStage(RuntimeError):
    pass


def run_stage(name: str, stage: Callable[[], None]) -> bool:
    try:
        stage()
    except SkipStage as error:
        print(f"SKIP {name}: {error}")
        return True
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()
        return False
    print(f"PASS {name}")
    return True


def stage_manifest() -> None:
    from modern_iopaint.model_manifest import load_model_manifest

    record = load_model_manifest().get(MODEL_NAME)
    if record.filename("int4", "r32", 0) != (
        "svdq-int4_r32-flux.1-fill-dev.safetensors"
    ):
        raise AssertionError("Unexpected FLUX int4 transformer filename")
    if record.filename("fp4", "r32", 0) != (
        "svdq-fp4_r32-flux.1-fill-dev.safetensors"
    ):
        raise AssertionError("Unexpected FLUX fp4 transformer filename")
    required_exclusions = {
        "transformer/*",
        "flux1-fill-dev.safetensors",
    }
    if not required_exclusions.issubset(record.base.ignore_patterns):
        raise AssertionError("FLUX base snapshot exclusions are incomplete")
    t5 = record.optional_components.get("nunchaku_t5")
    if t5 is None or t5.allow_patterns != (
        "awq-int4-flux.1-t5xxl.safetensors",
    ):
        raise AssertionError("Optional nunchaku-t5 component is missing")
    if not record.base.gated or record.gated:
        raise AssertionError("Only the BFL base component should be gated")
    print(
        f"transformer~{record.transformer_approx_download_size_bytes / GIB:.1f}GiB, "
        f"base~{record.base.approx_download_size_bytes / GIB:.1f}GiB, "
        f"nunchaku-t5~{t5.approx_download_size_bytes / GIB:.1f}GiB"
    )


def require_cuda():
    try:
        import torch
        from nunchaku import NunchakuFluxTransformer2dModel  # noqa: F401
    except Exception as error:
        raise SkipStage(f"Nunchaku 1.2.1 is unavailable: {error}") from error
    if not torch.cuda.is_available():
        raise SkipStage("CUDA is unavailable")
    return torch


def require_cached_artifacts(precision: str, *, require_t5: bool = False):
    from modern_iopaint.download import resolve_manifest_model_artifacts

    try:
        artifacts = resolve_manifest_model_artifacts(
            MODEL_NAME,
            precision=precision,
            rank="r32",
            lightning_steps=0,
        )
    except Exception as error:
        raise SkipStage(
            "FLUX Fill weights/base components are not fully cached or gated "
            f"access has not been completed: {error}"
        ) from error
    t5_path = artifacts.optional_component_paths.get("nunchaku_t5")
    if require_t5 and (t5_path is None or not t5_path.is_file()):
        raise SkipStage("the optional nunchaku-t5 int4 component is not cached")
    return artifacts


def make_test_inputs(size: int = 1024):
    import numpy as np
    from PIL import Image, ImageDraw

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(40, 210, size, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(210, 55, size, dtype=np.uint8)[:, None]
    image[:, :, 2] = 120
    mask = Image.new("L", (size, size), 0)
    margin = size // 4
    ImageDraw.Draw(mask).rectangle(
        (margin, margin, size - margin - 1, size - margin - 1), fill=255
    )
    return image, np.asarray(mask).copy()


def run_flux_inpaint(profile: str, precision: str) -> None:
    import numpy as np

    torch = require_cuda()
    require_cached_artifacts(
        precision,
        require_t5=profile == "conservative",
    )
    from modern_iopaint.model_manager import ModelManager
    from modern_iopaint.schema import HDStrategy, InpaintRequest

    image, mask = make_test_inputs()
    manager = ModelManager(
        name=MODEL_NAME,
        device=torch.device("cuda"),
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        local_files_only=True,
        enable_controlnet=False,
        qwen_precision="auto",
        qwen_rank="r32",
        qwen_lightning_steps=8,
        flux_precision=precision,
        runtime_profile=profile,
    )
    try:
        if manager.model.runtime.profile.value != profile:
            raise AssertionError(
                f"Expected {profile} profile, got {manager.model.runtime.profile.value}"
            )
        if profile == "conservative" and not manager.model.uses_nunchaku_t5:
            raise AssertionError("Conservative profile did not load int4 T5-XXL")
        request = InpaintRequest(
            hd_strategy=HDStrategy.ORIGINAL,
            prompt="a polished blue ceramic tile with subtle highlights",
            negative_prompt="this must be ignored by FLUX Fill",
            sd_steps=8,
            sd_guidance_scale=30.0,
            sd_strength=1.0,
            sd_mask_blur=0,
            sd_keep_unmasked_area=True,
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = manager(image, mask, request)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        print(
            f"{profile}: elapsed={elapsed:.2f}s, "
            f"peak_allocated={peak_allocated / GIB:.2f}GiB, "
            f"peak_reserved={peak_reserved / GIB:.2f}GiB"
        )
        if output.shape != (1024, 1024, 3) or output.dtype != np.uint8:
            raise AssertionError(
                f"Unexpected output: shape={output.shape}, dtype={output.dtype}"
            )
    finally:
        manager.unload()


def stage_gated_error(precision: str) -> None:
    from modern_iopaint.download import (
        FluxAccessError,
        _snapshot_download,
        is_manifest_model_downloaded,
    )
    from modern_iopaint.model_manifest import load_model_manifest

    if is_manifest_model_downloaded(
        MODEL_NAME,
        precision=precision,
        rank="r32",
        lightning_steps=0,
    ):
        raise SkipStage("FLUX weights are already cached; gated failure is not reproducible")

    record = load_model_manifest().get(MODEL_NAME)
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="modern-iopaint-p3-") as cache:
            try:
                _snapshot_download(
                    record.base,
                    Path(cache),
                    allow_patterns=record.base.allow_patterns,
                    local_files_only=True,
                )
            except FluxAccessError as error:
                message = str(error)
                required_text = (
                    "create or log into a Hugging Face account",
                    "accept the FLUX.1-dev license",
                    "HF_TOKEN",
                    "will not bypass repository gating",
                )
                missing = [text for text in required_text if text not in message]
                if missing:
                    raise AssertionError(
                        f"Structured gated error is missing text: {missing}"
                    ) from error
            else:
                raise AssertionError("Expected FluxAccessError for an empty offline cache")
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision", choices=("auto", "int4", "fp4"), default="auto"
    )
    args = parser.parse_args()
    results = [
        run_stage("(a) manifest", stage_manifest),
        run_stage("(b) fast masked inpaint", lambda: run_flux_inpaint("fast", args.precision)),
        run_stage(
            "(c) conservative masked inpaint",
            lambda: run_flux_inpaint("conservative", args.precision),
        ),
        run_stage("(d) gated error", lambda: stage_gated_error(args.precision)),
    ]
    print("PASS smoke-p3" if all(results) else "FAIL smoke-p3")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
