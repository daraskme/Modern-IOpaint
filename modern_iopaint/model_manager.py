from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from modern_iopaint.download import scan_models
from modern_iopaint.helper import switch_mps_device
from modern_iopaint.model import ControlNet, SD, SDXL, models
from modern_iopaint.model.utils import is_local_files_only, torch_gc
from modern_iopaint.schema import InpaintRequest, ModelInfo, ModelType


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.available_models: Dict[str, ModelInfo] = {}
        self.scan_models()

        self.enable_controlnet = kwargs.get("enable_controlnet", False)
        controlnet_method = kwargs.get("controlnet_method")
        if (
            controlnet_method is None
            and name in self.available_models
            and self.available_models[name].support_controlnet
            and self.available_models[name].controlnets
        ):
            controlnet_method = self.available_models[name].controlnets[0]
        self.controlnet_method = controlnet_method

        self.model = self.init_model(name, device, **kwargs)

    @property
    def current_model(self) -> ModelInfo:
        return self.available_models[self.name]

    def init_model(self, name: str, device, **kwargs):
        logger.info(f"Loading model: {name}")
        if name not in self.available_models:
            raise NotImplementedError(
                f"Unsupported model: {name}. Available models: {list(self.available_models.keys())}"
            )

        model_info = self.available_models[name]
        kwargs = {
            **kwargs,
            "model_info": model_info,
            "enable_controlnet": self.enable_controlnet,
            "controlnet_method": self.controlnet_method,
        }

        if model_info.support_controlnet and self.enable_controlnet:
            if self.controlnet_method not in model_info.controlnets:
                raise ValueError(
                    f"Unsupported ControlNet method {self.controlnet_method!r} for "
                    f"{name}. Supported methods: {model_info.controlnets}"
                )
            return ControlNet(device, **kwargs)

        if model_info.name in models:
            return models[name](device, **kwargs)

        if model_info.model_type in [
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SD,
        ]:
            return SD(device, **kwargs)

        if model_info.model_type in [
            ModelType.DIFFUSERS_SDXL_INPAINT,
            ModelType.DIFFUSERS_SDXL,
        ]:
            return SDXL(device, **kwargs)

        raise NotImplementedError(f"Unsupported model: {name}")

    @torch.inference_mode()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1] 255 means area to repaint
            config: inpainting request

        Returns:
            BGR uint8 image
        """
        controlnet_changed = config.enable_controlnet != self.enable_controlnet
        controlnet_method_changed = (
            config.enable_controlnet
            and config.controlnet_method != self.controlnet_method
        )
        if controlnet_changed or controlnet_method_changed:
            self.switch_controlnet_method(config)

        self.enable_disable_lcm_lora(config)
        return self.model(image, mask, config).astype(np.uint8)

    def scan_models(self) -> List[ModelInfo]:
        available_models = scan_models()
        self.available_models = {it.name: it for it in available_models}
        return available_models

    def switch(self, new_name: str):
        if new_name == self.name:
            return

        old_name = self.name
        old_controlnet_method = self.controlnet_method
        self.name = new_name

        if (
            self.available_models[new_name].support_controlnet
            and self.controlnet_method
            not in self.available_models[new_name].controlnets
        ):
            self.controlnet_method = self.available_models[new_name].controlnets[0]
        try:
            del self.model
            torch_gc()

            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
        except Exception as error:
            self.name = old_name
            self.controlnet_method = old_controlnet_method
            logger.info(f"Switch model from {old_name} to {new_name} failed, rollback")
            self.model = self.init_model(
                old_name, switch_mps_device(old_name, self.device), **self.kwargs
            )
            raise error

    def switch_controlnet_method(self, config: InpaintRequest):
        model_info = self.available_models[self.name]
        if not model_info.support_controlnet:
            if config.enable_controlnet:
                raise ValueError(f"Model {self.name} does not support ControlNet")
            return

        if config.enable_controlnet and config.controlnet_method not in model_info.controlnets:
            raise ValueError(
                f"Unsupported ControlNet method {config.controlnet_method!r} for "
                f"{self.name}. Supported methods: {model_info.controlnets}"
            )

        if (
            self.enable_controlnet
            and config.enable_controlnet
            and self.controlnet_method != config.controlnet_method
        ):
            old_controlnet_method = self.controlnet_method
            self.controlnet_method = config.controlnet_method
            self.model.switch_controlnet_method(config.controlnet_method)
            logger.info(
                f"Switch ControlNet method from {old_controlnet_method} "
                f"to {config.controlnet_method}"
            )
            return

        if self.enable_controlnet == config.enable_controlnet:
            return

        self.enable_controlnet = config.enable_controlnet
        self.controlnet_method = config.controlnet_method

        pipe_components = {
            "vae": self.model.model.vae,
            "text_encoder": self.model.model.text_encoder,
            "unet": self.model.model.unet,
        }
        if hasattr(self.model.model, "text_encoder_2"):
            pipe_components["text_encoder_2"] = self.model.model.text_encoder_2

        self.model = self.init_model(
            self.name,
            switch_mps_device(self.name, self.device),
            pipe_components=pipe_components,
            **self.kwargs,
        )
        if config.enable_controlnet:
            logger.info(f"Enable ControlNet: {config.controlnet_method}")
        else:
            logger.info("Disable ControlNet")

    def enable_disable_lcm_lora(self, config: InpaintRequest):
        if not self.available_models[self.name].support_lcm_lora:
            return

        pipe = self.model.model
        get_list_adapters = getattr(pipe, "get_list_adapters", None)
        if callable(get_list_adapters):
            lcm_lora_loaded = bool(get_list_adapters())
        else:
            lcm_lora_loaded = bool(getattr(pipe, "peft_config", {}))

        if config.sd_lcm_lora:
            if not lcm_lora_loaded:
                logger.info("Load LCM LoRA")
                pipe.load_lora_weights(
                    self.model.lcm_lora_id,
                    weight_name="pytorch_lora_weights.safetensors",
                    local_files_only=is_local_files_only(),
                )
            else:
                logger.info("Enable LCM LoRA")
                pipe.enable_lora()
        elif lcm_lora_loaded:
            logger.info("Disable LCM LoRA")
            pipe.disable_lora()

