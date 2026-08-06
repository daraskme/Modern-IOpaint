"""Phase P1 smoke test for the retained Modern-IOPaint model paths.

The script downloads missing model weights through the normal application helpers.
Run the LaMa CPU smoke with::

    python scripts/smoke_p1.py

Pass ``--gpu`` to additionally run the default SD 1.5 inpaint model for three
steps when CUDA is available.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable


def run_stage(name: str, stage: Callable[[], None]) -> bool:
    try:
        stage()
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()
        return False
    print(f"PASS {name}")
    return True


def make_test_inputs(size: int):
    import numpy as np
    from PIL import Image, ImageDraw

    image_array = np.zeros((size, size, 3), dtype=np.uint8)
    image_array[:, :, 0] = np.linspace(24, 224, size, dtype=np.uint8)[None, :]
    image_array[:, :, 1] = np.linspace(224, 24, size, dtype=np.uint8)[:, None]
    image_array[:, :, 2] = 128
    image = Image.fromarray(image_array, mode="RGB")

    mask = Image.new("L", (size, size), 0)
    inset = size // 4
    ImageDraw.Draw(mask).rectangle(
        (inset, inset, size - inset - 1, size - inset - 1), fill=255
    )
    return np.asarray(image).copy(), np.asarray(mask).copy()


def ensure_model(model_name: str) -> None:
    from modern_iopaint.download import cli_download_model, scan_models

    if model_name not in {model.name for model in scan_models()}:
        cli_download_model(model_name)


def make_config(**overrides):
    from modern_iopaint.schema import HDStrategy, InpaintRequest, SDSampler

    values = {
        "hd_strategy": HDStrategy.ORIGINAL,
        "sd_sampler": SDSampler.uni_pc,
        "sd_keep_unmasked_area": True,
    }
    values.update(overrides)
    return InpaintRequest(**values)


def smoke_lama(inputs: dict[str, object]) -> None:
    import numpy as np
    import torch

    from modern_iopaint.model_manager import ModelManager

    ensure_model("lama")
    manager = ModelManager(name="lama", device=torch.device("cpu"))
    output = manager(inputs["image_512"], inputs["mask_512"], make_config())
    if output.shape != (512, 512, 3):
        raise AssertionError(f"Unexpected LaMa output shape: {output.shape}")
    if output.dtype != np.uint8:
        raise AssertionError(f"Unexpected LaMa output dtype: {output.dtype}")


def smoke_sd15_gpu(inputs: dict[str, object], requested: bool) -> None:
    import numpy as np
    import torch

    if not requested:
        print("PASS gpu-sd15 (skipped; pass --gpu to request it)")
        return
    if not torch.cuda.is_available():
        print("PASS gpu-sd15 (skipped; CUDA is not available)")
        return

    from modern_iopaint.model_manager import ModelManager

    model_name = "runwayml/stable-diffusion-inpainting"
    ensure_model(model_name)
    manager = ModelManager(
        name=model_name,
        device=torch.device("cuda"),
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        local_files_only=False,
        enable_controlnet=False,
    )
    # The production wrapper pads small inputs to 512. Override the inherited
    # minimum only for this fast compatibility smoke so the pipeline sees 256x256.
    manager.model.min_size = 256
    config = make_config(
        prompt="a small red fox in a forest",
        negative_prompt="blurry, low quality",
        sd_steps=3,
        sd_strength=1.0,
        sd_guidance_scale=7.5,
        sd_scale=1.0,
    )
    output = manager(inputs["image_256"], inputs["mask_256"], config)
    if output.shape != (256, 256, 3):
        raise AssertionError(f"Unexpected SD 1.5 output shape: {output.shape}")
    if output.dtype != np.uint8:
        raise AssertionError(f"Unexpected SD 1.5 output dtype: {output.dtype}")
    print("PASS gpu-sd15")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also run a three-step SD 1.5 inpaint smoke when CUDA is available",
    )
    args = parser.parse_args()

    inputs: dict[str, object] = {}

    def generate_inputs() -> None:
        inputs["image_512"], inputs["mask_512"] = make_test_inputs(512)
        inputs["image_256"], inputs["mask_256"] = make_test_inputs(256)

    results = [
        run_stage("generate-inputs", generate_inputs),
        run_stage("lama-cpu", lambda: smoke_lama(inputs)),
    ]

    try:
        smoke_sd15_gpu(inputs, args.gpu)
    except Exception:
        print("FAIL gpu-sd15")
        traceback.print_exc()
        results.append(False)
    else:
        results.append(True)

    print("PASS smoke-p1" if all(results) else "FAIL smoke-p1")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
