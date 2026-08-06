from __future__ import annotations

from typing import Any

import cv2
import PIL.Image
import torch
from loguru import logger

from modern_iopaint.const import FLUX_FILL_NAME
from modern_iopaint.runtime_profile import RuntimeProfile, select_runtime_profile
from modern_iopaint.schema import InpaintRequest

from .base import DiffusionInpaintModel


class FluxFill(DiffusionInpaintModel):
    """FLUX.1-Fill-dev inpainting with a quantized Nunchaku transformer."""

    name = FLUX_FILL_NAME
    pad_mod = 16
    min_size = 1024

    def init_model(self, device: torch.device, **kwargs):
        if str(device).split(":", 1)[0] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FLUX Fill/Nunchaku requires an available CUDA device")

        try:
            from diffusers import FluxFillPipeline
            from nunchaku import (
                NunchakuFluxTransformer2dModel,
                NunchakuT5EncoderModel,
            )
            from nunchaku.utils import is_turing
        except Exception as error:
            raise RuntimeError(
                "FLUX Fill requires nunchaku==1.2.1 and diffusers==0.36.0. "
                "Install Nunchaku separately for this CUDA environment."
            ) from error

        from modern_iopaint.download import resolve_manifest_model_artifacts

        artifacts = resolve_manifest_model_artifacts(
            self.name,
            precision=kwargs.get("flux_precision", "auto"),
            rank="r32",
            lightning_steps=0,
            cache_dir=kwargs.get("model_cache_dir"),
        )
        self.precision = artifacts.precision
        self.rank = artifacts.rank
        self.runtime = select_runtime_profile(device, kwargs.get("runtime_profile"))
        self.callback = kwargs.get("callback")
        torch_dtype = torch.float16 if is_turing() else torch.bfloat16
        transformer_offload = self.runtime.profile is RuntimeProfile.conservative

        logger.info(
            "Loading {} transformer: {} ({}, {}, profile={}, offload={})",
            self.name,
            artifacts.transformer_path,
            self.precision,
            self.rank,
            self.runtime.profile.value,
            transformer_offload,
        )
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            str(artifacts.transformer_path),
            torch_dtype=torch_dtype,
            device=device,
            precision=self.precision,
            offload=transformer_offload,
        )

        pipeline_kwargs: dict[str, Any] = {
            "transformer": transformer,
            "torch_dtype": torch_dtype,
            "local_files_only": True,
        }
        self.uses_nunchaku_t5 = False
        if self.runtime.profile is RuntimeProfile.conservative:
            t5_path = artifacts.optional_component_paths.get("nunchaku_t5")
            if t5_path is not None and t5_path.is_file():
                logger.info("Loading conservative-profile int4 T5-XXL: {}", t5_path)
                pipeline_kwargs["text_encoder_2"] = (
                    NunchakuT5EncoderModel.from_pretrained(
                        str(t5_path),
                        torch_dtype=torch_dtype,
                        device=device,
                    )
                )
                self.uses_nunchaku_t5 = True
            else:
                logger.warning(
                    "The optional nunchaku-t5 component is not cached; the "
                    "conservative profile will use the base bf16 T5-XXL. Run "
                    "`modern-iopaint download --model {}` to add the int4 T5.",
                    self.name,
                )

        self.model = FluxFillPipeline.from_pretrained(
            str(artifacts.base_path),
            **pipeline_kwargs,
        )
        self._apply_runtime_profile()

    def _apply_runtime_profile(self) -> None:
        if self.runtime.profile is RuntimeProfile.conservative:
            # Nunchaku Flux offload is selected during from_pretrained. Its
            # installed reference path then applies Diffusers sequential
            # offload to the assembled pipeline.
            self.model.enable_sequential_cpu_offload()
            self.model.enable_vae_tiling()
            return

        self.model.enable_model_cpu_offload()
        if self.runtime.profile is RuntimeProfile.balanced:
            self.model.enable_vae_tiling()

    @staticmethod
    def _mask_image(mask) -> PIL.Image.Image:
        if mask.ndim == 3:
            mask = mask[:, :, -1]
        return PIL.Image.fromarray(mask.astype("uint8"), mode="L")

    @staticmethod
    def _inference_parameters(config: InpaintRequest) -> dict[str, Any]:
        # The shared request schema predates FLUX. Preserve explicit values,
        # while translating its legacy defaults to Fill-dev conventions.
        steps = 28 if config.sd_steps == 50 else config.sd_steps
        guidance_scale = 30.0 if config.sd_guidance_scale == 7.5 else config.sd_guidance_scale
        return {
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }

    def forward(self, image, mask, config: InpaintRequest):
        img_h, img_w = image.shape[:2]
        generator = torch.Generator(device=self.device).manual_seed(config.sd_seed)
        output = self.model(
            image=PIL.Image.fromarray(image),
            mask_image=self._mask_image(mask),
            prompt=config.prompt,
            strength=config.sd_strength,
            height=img_h,
            width=img_w,
            output_type="np",
            generator=generator,
            callback_on_step_end=self.callback,
            **self._inference_parameters(config),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    @staticmethod
    def is_downloaded() -> bool:
        from modern_iopaint.download import is_manifest_model_downloaded

        return is_manifest_model_downloaded(
            FLUX_FILL_NAME,
            rank="r32",
            lightning_steps=0,
        )

    @staticmethod
    def download():
        from modern_iopaint.download import cli_download_model

        return cli_download_model(FLUX_FILL_NAME)
