from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


RUNTIME_PROFILE_ENV = "MODERN_IOPAINT_RUNTIME_PROFILE"
GIB = 1024**3


class RuntimeProfile(str, Enum):
    auto = "auto"
    fast = "fast"
    balanced = "balanced"
    conservative = "conservative"


@dataclass(frozen=True)
class RuntimeProfileSelection:
    profile: RuntimeProfile
    free_vram_bytes: int
    total_vram_bytes: int
    total_ram_bytes: int
    num_blocks_on_gpu: Optional[int] = None

    @property
    def free_vram_gib(self) -> float:
        return self.free_vram_bytes / GIB

    @property
    def total_vram_gib(self) -> float:
        return self.total_vram_bytes / GIB

    @property
    def total_ram_gib(self) -> float:
        return self.total_ram_bytes / GIB


def _total_system_ram() -> int:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * physical_pages
        except (OSError, ValueError, TypeError):
            pass

    logger.warning("Unable to detect total system RAM")
    return 0


def _normalize_override(override: Optional[str]) -> RuntimeProfile:
    env_override = os.getenv(RUNTIME_PROFILE_ENV)
    if env_override and (not override or override == RuntimeProfile.auto.value):
        override = env_override
    raw = override or RuntimeProfile.auto.value
    if isinstance(raw, RuntimeProfile):
        return raw
    try:
        return RuntimeProfile(str(raw).lower())
    except ValueError as error:
        choices = ", ".join(profile.value for profile in RuntimeProfile)
        raise ValueError(
            f"Invalid runtime profile {raw!r}; choose one of: {choices}"
        ) from error


def _conservative_blocks(free_vram_bytes: int) -> int:
    """Keep a small, monotonic number of transformer blocks resident on GPU."""

    free_gib = free_vram_bytes / GIB
    headroom_adjusted = max(0.0, free_gib - 5.0)
    return max(1, min(8, int(headroom_adjusted / 1.5) + 1))


def select_runtime_profile(device, override: Optional[str] = None) -> RuntimeProfileSelection:
    import torch

    if not torch.cuda.is_available() or str(device).split(":", 1)[0] != "cuda":
        raise RuntimeError("Nunchaku backends require an available CUDA device")

    device_index = getattr(device, "index", None)
    if device_index is None:
        device_index = torch.cuda.current_device()
    free_vram, total_vram = torch.cuda.mem_get_info(device_index)
    total_ram = _total_system_ram()
    requested = _normalize_override(override)

    if requested is RuntimeProfile.auto:
        free_gib = free_vram / GIB
        if free_gib >= 24.0:
            selected = RuntimeProfile.fast
        elif free_gib >= 19.0:
            selected = RuntimeProfile.balanced
        else:
            selected = RuntimeProfile.conservative
    else:
        selected = requested

    blocks = (
        _conservative_blocks(free_vram)
        if selected is RuntimeProfile.conservative
        else None
    )
    selection = RuntimeProfileSelection(
        profile=selected,
        free_vram_bytes=int(free_vram),
        total_vram_bytes=int(total_vram),
        total_ram_bytes=total_ram,
        num_blocks_on_gpu=blocks,
    )
    override_text = (
        "auto-selected" if requested is RuntimeProfile.auto else "explicit override"
    )
    block_text = (
        f", transformer blocks on GPU={blocks}" if blocks is not None else ""
    )
    logger.info(
        "Nunchaku runtime profile: {} ({}; free VRAM={:.2f} GiB / {:.2f} GiB, "
        "system RAM={:.2f} GiB{})",
        selection.profile.value,
        override_text,
        selection.free_vram_gib,
        selection.total_vram_gib,
        selection.total_ram_gib,
        block_text,
    )
    return selection
