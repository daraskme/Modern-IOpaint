from __future__ import annotations

from typing import Any, Type

from modern_iopaint.const import QWEN_IMAGE_EDIT_NAME

from .qwen_image import QwenImage


class QwenImageEdit(QwenImage):
    name = QWEN_IMAGE_EDIT_NAME

    def _pipeline_class(self) -> Type[Any]:
        from diffusers import QwenImageEditInpaintPipeline

        return QwenImageEditInpaintPipeline

    @staticmethod
    def is_downloaded() -> bool:
        from modern_iopaint.download import is_manifest_model_downloaded

        return is_manifest_model_downloaded(QWEN_IMAGE_EDIT_NAME)

    @staticmethod
    def download():
        from modern_iopaint.download import cli_download_model

        return cli_download_model(QWEN_IMAGE_EDIT_NAME)
