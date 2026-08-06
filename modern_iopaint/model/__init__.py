from .controlnet import ControlNet
from .fcf import FcF
from .instruct_pix2pix import InstructPix2Pix
from .kandinsky import Kandinsky22
from .lama import LaMa, AnimeLaMa
from .ldm import LDM
from .manga import Manga
from .mat import MAT
from .mi_gan import MIGAN
from .opencv2 import OpenCV2
from .paint_by_example import PaintByExample
from .qwen_image import QwenImage
from .qwen_image_edit import QwenImageEdit
from .sd import SD15, SD2, Anything4, RealisticVision14, SD
from .sdxl import SDXL
from .zits import ZITS

try:
    # Importing the platform-specific extension is the registry capability
    # check. FLUX Fill is not registered when Nunchaku cannot be loaded.
    from nunchaku import NunchakuFluxTransformer2dModel as _NunchakuFlux  # noqa: F401

    from .flux_fill import FluxFill
except Exception:
    FluxFill = None

models = {
    LaMa.name: LaMa,
    AnimeLaMa.name: AnimeLaMa,
    LDM.name: LDM,
    ZITS.name: ZITS,
    MAT.name: MAT,
    FcF.name: FcF,
    OpenCV2.name: OpenCV2,
    Manga.name: Manga,
    MIGAN.name: MIGAN,
    SD15.name: SD15,
    Anything4.name: Anything4,
    RealisticVision14.name: RealisticVision14,
    SD2.name: SD2,
    PaintByExample.name: PaintByExample,
    QwenImage.name: QwenImage,
    QwenImageEdit.name: QwenImageEdit,
    InstructPix2Pix.name: InstructPix2Pix,
    Kandinsky22.name: Kandinsky22,
    SDXL.name: SDXL,
}

if FluxFill is not None:
    models[FluxFill.name] = FluxFill
