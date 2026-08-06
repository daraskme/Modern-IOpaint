from pathlib import Path
from typing import Dict

CURRENT_DIR = Path(__file__).parent.absolute()


def get_config_files() -> Dict[str, Path]:
    """
    - `v1`: Config file for Stable Diffusion v1
    - `v2`: Config file for Stable Diffusion v2
    - `xl`: Config file for Stable Diffusion XL
    - `xl_refiner`: Config file for Stable Diffusion XL Refiner
    """
    return {
        "v1": CURRENT_DIR / "v1-inference.yaml",
        "v2": CURRENT_DIR / "v2-inference-v.yaml",
        "xl": CURRENT_DIR / "sd_xl_base.yaml",
        "xl_refiner": CURRENT_DIR / "sd_xl_refiner.yaml",
    }


def load_original_config(name: str) -> str:
    """Return a legacy Stable Diffusion YAML path for Diffusers' single-file API."""

    # Diffusers 0.36 resolves ``original_config`` as a filesystem path or URL
    # before parsing the YAML. Passing an already-parsed dict reaches
    # os.path.isfile() inside Diffusers and fails before the checkpoint loads.
    return str(get_config_files()[name])
