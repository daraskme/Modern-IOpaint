from __future__ import annotations

from typing import Any, Type

import cv2
import PIL.Image
import torch
from loguru import logger

from modern_iopaint.const import QWEN_IMAGE_NAME
from modern_iopaint.runtime_profile import RuntimeProfile, select_runtime_profile
from modern_iopaint.schema import InpaintRequest, ModelCategory

from .base import DiffusionInpaintModel


class QwenImage(DiffusionInpaintModel):
    name = QWEN_IMAGE_NAME
    category = ModelCategory.INPAINT_GENERAL
    pad_mod = 16
    min_size = 1024

    def _pipeline_class(self) -> Type[Any]:
        from diffusers import QwenImageInpaintPipeline

        return QwenImageInpaintPipeline

    def init_model(self, device: torch.device, **kwargs):
        if str(device).split(":", 1)[0] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Qwen/Nunchaku backends require an available CUDA device")

        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel
        except Exception as error:
            raise RuntimeError(
                "Qwen models require nunchaku==1.2.1, installed separately in "
                "the Modern-IOPaint environment. LaMa and Stable Diffusion do "
                "not require Nunchaku."
            ) from error

        precision = kwargs.get("qwen_precision")
        rank = kwargs.get("qwen_rank")
        lightning_steps = kwargs.get("qwen_lightning_steps")
        from modern_iopaint.download import resolve_manifest_model_artifacts

        artifacts = resolve_manifest_model_artifacts(
            self.name,
            precision=precision,
            rank=rank,
            lightning_steps=lightning_steps,
            cache_dir=kwargs.get("model_cache_dir"),
        )
        self.precision = artifacts.precision
        self.rank = artifacts.rank
        self.lightning_steps = artifacts.lightning_steps
        self.runtime = select_runtime_profile(device, kwargs.get("runtime_profile"))
        logger.info(
            "Loading {} transformer: {} ({}, {}, lightning={})",
            self.name,
            artifacts.transformer_path,
            self.precision,
            self.rank,
            self.lightning_steps or "none",
        )

        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            str(artifacts.transformer_path)
        )
        pipeline_class = self._pipeline_class()
        self.model = pipeline_class.from_pretrained(
            str(artifacts.base_path),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self._apply_runtime_profile(transformer, device)
        self.callback = kwargs.get("callback")

    def _enable_vae_tiling(self) -> None:
        enable_pipeline_tiling = getattr(self.model, "enable_vae_tiling", None)
        if callable(enable_pipeline_tiling):
            enable_pipeline_tiling()
            return
        vae = getattr(self.model, "vae", None)
        enable_vae_tiling = getattr(vae, "enable_tiling", None)
        if callable(enable_vae_tiling):
            enable_vae_tiling()

    def _apply_runtime_profile(self, transformer, device: torch.device) -> None:
        if self.runtime.profile is RuntimeProfile.conservative:
            transformer.set_offload(
                True,
                use_pin_memory=False,
                num_blocks_on_gpu=self.runtime.num_blocks_on_gpu,
            )
            # DiffusionPipeline defines this as an empty class-level list. Always
            # assign an instance list: ``... or []`` followed only by append would
            # mutate a detached temporary and let Accelerate meta-offload the
            # Nunchaku transformer.
            excluded = list(
                getattr(self.model, "_exclude_from_cpu_offload", ()) or ()
            )
            if "transformer" not in excluded:
                excluded.append("transformer")
            self.model._exclude_from_cpu_offload = excluded
            self.model.enable_sequential_cpu_offload()
            self._enable_vae_tiling()
            return

        self.model.enable_model_cpu_offload()
        if self.runtime.profile is RuntimeProfile.balanced:
            self._enable_vae_tiling()

    def _inference_parameters(self, config: InpaintRequest) -> dict[str, Any]:
        steps = config.sd_steps
        if self.lightning_steps and steps == 50:
            steps = self.lightning_steps

        if self.lightning_steps:
            guidance_scale = config.sd_guidance_scale
            if guidance_scale == 7.5:
                guidance_scale = 1.0
            return {
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": 1.0,
                "negative_prompt": None,
            }

        return {
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "true_cfg_scale": config.sd_guidance_scale,
            "negative_prompt": config.negative_prompt or None,
        }

    def forward(self, image, mask, config: InpaintRequest):
        img_h, img_w = image.shape[:2]
        generator = torch.Generator(device=self.device).manual_seed(config.sd_seed)
        output = self.model(
            image=PIL.Image.fromarray(image),
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
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

        return is_manifest_model_downloaded(QWEN_IMAGE_NAME)

    @staticmethod
    def download():
        from modern_iopaint.download import cli_download_model

        return cli_download_model(QWEN_IMAGE_NAME)
