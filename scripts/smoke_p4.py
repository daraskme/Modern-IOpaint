"""Phase P4 illustration/anime support smoke test.

The AnimeLaMa stage uses CPU and downloads its normal upstream checkpoint when
it is not cached. The local SDXL stage only runs with ``--checkpoint PATH``.
The Qwen stage uses the P2 CUDA/Nunchaku options and can be disabled with
``--skip-qwen``. This script does not download Qwen or local checkpoint files.
"""

from __future__ import annotations

import argparse
import gc
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


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


def make_cel_inputs(size: int):
    """Create flat color regions and hard black outlines with Pillow."""

    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "#8fd3ff")
    draw = ImageDraw.Draw(image)
    outline = max(5, size // 96)

    draw.rectangle(
        (0, size * 2 // 3, size, size),
        fill="#72c66b",
        outline="#111111",
        width=outline,
    )
    draw.polygon(
        (
            (0, size * 2 // 3),
            (size // 4, size // 3),
            (size // 2, size * 2 // 3),
        ),
        fill="#7766bb",
        outline="#111111",
    )
    draw.polygon(
        (
            (size // 3, size * 2 // 3),
            (size * 3 // 4, size // 4),
            (size, size * 2 // 3),
        ),
        fill="#a98bd4",
        outline="#111111",
    )

    head = (size // 3, size // 6, size * 2 // 3, size // 2)
    draw.ellipse(head, fill="#ffd2ae", outline="#111111", width=outline)
    draw.polygon(
        (
            (size // 3, size // 3),
            (size * 2 // 5, size // 7),
            (size // 2, size // 5),
            (size * 3 // 5, size // 7),
            (size * 2 // 3, size // 3),
            (size * 3 // 5, size // 4),
            (size // 2, size // 3),
            (size * 2 // 5, size // 4),
        ),
        fill="#26355f",
        outline="#111111",
    )
    eye_radius = max(4, size // 90)
    for eye_x in (size * 9 // 20, size * 11 // 20):
        eye_y = size // 3
        draw.ellipse(
            (
                eye_x - eye_radius,
                eye_y - eye_radius,
                eye_x + eye_radius,
                eye_y + eye_radius,
            ),
            fill="#111111",
        )
    draw.rounded_rectangle(
        (size * 3 // 8, size * 5 // 11, size * 5 // 8, size * 17 // 20),
        radius=size // 24,
        fill="#ef596f",
        outline="#111111",
        width=outline,
    )

    # The masked yellow badge is the content erased/repainted by all stages.
    badge = (
        size * 9 // 20,
        size * 11 // 20,
        size * 11 // 20,
        size * 13 // 20,
    )
    draw.ellipse(badge, fill="#ffd84d", outline="#111111", width=outline)
    mask_image = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    mask_margin = size // 64
    mask_draw.ellipse(
        (
            badge[0] - mask_margin,
            badge[1] - mask_margin,
            badge[2] + mask_margin,
            badge[3] + mask_margin,
        ),
        fill=255,
    )
    return np.asarray(image).copy(), np.asarray(mask_image).copy()


def make_request(*, steps: int, guidance: float, prompt: str):
    from modern_iopaint.schema import HDStrategy, InpaintRequest, SDSampler

    return InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        prompt=prompt,
        negative_prompt="photorealistic, blurry, soft outlines",
        sd_sampler=SDSampler.euler,
        sd_steps=steps,
        sd_guidance_scale=guidance,
        sd_strength=1.0,
        sd_mask_blur=0,
        sd_keep_unmasked_area=True,
    )


def stage_anime_lama(shared: dict[str, Any]) -> None:
    import numpy as np
    import torch

    from modern_iopaint.download import cli_download_model
    from modern_iopaint.model.lama import AnimeLaMa
    from modern_iopaint.model_manager import ModelManager
    from modern_iopaint.schema import ModelCategory

    image, mask = make_cel_inputs(1024)
    shared["cel_image"] = image
    shared["cel_mask"] = mask
    if not AnimeLaMa.is_downloaded():
        cli_download_model("anime-lama")

    manager = ModelManager(name="anime-lama", device=torch.device("cpu"))
    try:
        if manager.current_model.category != ModelCategory.ERASE_ILLUSTRATION:
            raise AssertionError(
                f"Unexpected AnimeLaMa category: {manager.current_model.category}"
            )
        output = manager(
            image,
            mask,
            make_request(steps=2, guidance=1.0, prompt=""),
        )
        if output.shape != (1024, 1024, 3) or output.dtype != np.uint8:
            raise AssertionError(
                f"Unexpected AnimeLaMa output: shape={output.shape}, "
                f"dtype={output.dtype}"
            )
    finally:
        manager.unload()


def stage_local_checkpoint(checkpoint: Path | None) -> None:
    if checkpoint is None:
        raise SkipStage("--checkpoint PATH was not provided")

    import numpy as np
    import torch

    from modern_iopaint.download import get_sdxl_model_type
    from modern_iopaint.model.sdxl import SDXL
    from modern_iopaint.model_metadata import load_local_model_metadata
    from modern_iopaint.schema import ModelInfo, ModelType

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if checkpoint.suffix.lower() not in (".ckpt", ".safetensors"):
        raise ValueError("--checkpoint must be a .ckpt or .safetensors file")

    metadata = load_local_model_metadata(checkpoint)
    if not metadata.sidecar_exists:
        raise AssertionError(f"Sidecar is missing: {metadata.sidecar_path}")
    if metadata.prediction_type is None:
        raise AssertionError(
            f"Sidecar {metadata.sidecar_path} must set prediction_type"
        )

    model_type = get_sdxl_model_type(str(checkpoint))
    if model_type not in (
        ModelType.DIFFUSERS_SDXL,
        ModelType.DIFFUSERS_SDXL_INPAINT,
    ):
        raise AssertionError(f"Checkpoint was not detected as SDXL: {model_type}")

    model_info = ModelInfo(
        name=checkpoint.name,
        path=str(checkpoint),
        model_type=model_type,
        category=metadata.category,
        is_single_file_diffusers=True,
        prediction_type=metadata.prediction_type,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDXL(
        device,
        model_info=model_info,
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        local_files_only=True,
        enable_controlnet=False,
    )
    try:
        detected = model.model.scheduler.config.prediction_type
        if detected != metadata.prediction_type:
            raise AssertionError(
                f"Sidecar prediction_type was not applied: "
                f"expected {metadata.prediction_type}, got {detected}"
            )

        image, mask = make_cel_inputs(512)
        output = model(
            image,
            mask,
            make_request(
                steps=2,
                guidance=5.0,
                prompt="clean cel-shaded red jacket with crisp black line art",
            ),
        ).astype(np.uint8)
        if model.model.scheduler.config.prediction_type != metadata.prediction_type:
            raise AssertionError("Sampler replacement lost the prediction_type override")
        if output.shape != (512, 512, 3) or output.dtype != np.uint8:
            raise AssertionError(
                f"Unexpected local SDXL output: shape={output.shape}, "
                f"dtype={output.dtype}"
            )
    finally:
        model.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_qwen(shared: dict[str, Any], args: argparse.Namespace) -> None:
    if args.skip_qwen:
        raise SkipStage("--skip-qwen was passed")

    import numpy as np

    import smoke_p2

    torch = smoke_p2.require_cuda()
    smoke_p2.assert_variant_downloaded("qwen-image", args)
    from modern_iopaint.model_manager import ModelManager
    from modern_iopaint.schema import ModelCategory

    if "cel_image" not in shared:
        shared["cel_image"], shared["cel_mask"] = make_cel_inputs(1024)
    manager = ModelManager(
        name="qwen-image",
        device=torch.device("cuda"),
        **smoke_p2.manager_kwargs(args),
    )
    try:
        if manager.current_model.category != ModelCategory.INPAINT_GENERAL:
            raise AssertionError(
                f"Unexpected Qwen category: {manager.current_model.category}"
            )
        output = manager(
            shared["cel_image"],
            shared["cel_mask"],
            make_request(
                steps=args.steps,
                guidance=1.0,
                prompt=(
                    "replace the masked badge with a small blue ribbon, anime "
                    "cel shading, flat colors, crisp hard black outlines"
                ),
            ),
        )
        if output.shape != (1024, 1024, 3) or output.dtype != np.uint8:
            raise AssertionError(
                f"Unexpected Qwen output: shape={output.shape}, dtype={output.dtype}"
            )
    finally:
        manager.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local SDXL .ckpt/.safetensors file; requires a same-name JSON sidecar",
    )
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument(
        "--precision", choices=("auto", "int4", "fp4"), default="auto"
    )
    parser.add_argument("--rank", choices=("r32", "r128"), default="r32")
    parser.add_argument("--steps", choices=(4, 8), default=8, type=int)
    parser.add_argument(
        "--profile",
        choices=("auto", "fast", "balanced", "conservative"),
        default="auto",
    )
    args = parser.parse_args()
    shared: dict[str, Any] = {}

    results = [
        run_stage("(a) anime-lama-cpu-erase", lambda: stage_anime_lama(shared)),
        run_stage(
            "(b) local-sdxl-sidecar-inpaint",
            lambda: stage_local_checkpoint(args.checkpoint),
        ),
        run_stage("(c) qwen-anime-inpaint", lambda: stage_qwen(shared, args)),
    ]
    print("PASS smoke-p4" if all(results) else "FAIL smoke-p4")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
