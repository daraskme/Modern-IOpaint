"""Phase P2 Qwen/model-management smoke test.

Preconditions:
  * CUDA is available and nunchaku==1.2.1 is installed separately.
  * The selected qwen-image and qwen-image-edit variants were downloaded with
    ``modern-iopaint download`` (edit may be omitted with --skip-edit).
  * LaMa is downloaded for the model-switch stage.

This script does not download models. Its default is the lightning 8-step r32
variant with Nunchaku precision auto-detection.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any


GIB = 1024**3
MIB = 1024**2
CUDA_ALLOCATED_GROWTH_TOLERANCE = 256 * MIB
CUDA_RESERVED_GROWTH_TOLERANCE = 512 * MIB
RSS_GROWTH_TOLERANCE = 512 * MIB


def run_stage(name: str, stage: Callable[[], None]) -> bool:
    try:
        stage()
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()
        return False
    print(f"PASS {name}")
    return True


def make_test_inputs(size: int = 1024):
    import numpy as np
    from PIL import Image, ImageDraw

    image_array = np.zeros((size, size, 3), dtype=np.uint8)
    image_array[:, :, 0] = np.linspace(32, 224, size, dtype=np.uint8)[None, :]
    image_array[:, :, 1] = np.linspace(224, 48, size, dtype=np.uint8)[:, None]
    image_array[:, :, 2] = 112

    mask_image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_image)
    margin = size // 4
    draw.ellipse(
        (margin, margin, size - margin - 1, size - margin - 1),
        fill=255,
    )
    return image_array, np.asarray(mask_image).copy()


def manager_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "no_half": False,
        "low_mem": False,
        "cpu_offload": False,
        "disable_nsfw": True,
        "sd_cpu_textencoder": False,
        "local_files_only": True,
        "enable_controlnet": False,
        "qwen_precision": args.precision,
        "qwen_rank": args.rank,
        "qwen_lightning_steps": args.steps,
        "runtime_profile": args.profile,
    }


def make_request(steps: int, *, edit: bool = False):
    from modern_iopaint.schema import HDStrategy, InpaintRequest

    prompt = (
        "Replace the masked circle with a polished blue ceramic tile, preserving "
        "the surrounding image"
        if edit
        else "a polished blue ceramic tile with subtle highlights"
    )
    return InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        prompt=prompt,
        negative_prompt="",
        sd_steps=steps,
        sd_guidance_scale=1.0,
        sd_strength=1.0,
        sd_mask_blur=0,
        sd_keep_unmasked_area=True,
    )


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("P2 Qwen smoke requires CUDA")
    return torch


def process_rss_bytes() -> int:
    """Return current process resident memory without an optional dependency."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))

    # ru_maxrss is the best standard-library fallback on other POSIX systems.
    # It is a high-water mark rather than current RSS, so this makes the leak
    # check conservative on those platforms.
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def memory_snapshot(torch, label: str) -> dict[str, int]:
    torch.cuda.synchronize()
    snapshot = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "rss": process_rss_bytes(),
    }
    print(
        f"{label}: allocated={snapshot['allocated'] / GIB:.2f}GiB, "
        f"reserved={snapshot['reserved'] / GIB:.2f}GiB, "
        f"rss={snapshot['rss'] / GIB:.2f}GiB"
    )
    return snapshot


def assert_memory_stable(
    phase: str,
    earlier_cycle: dict[str, int],
    later_cycle: dict[str, int],
    earlier_cycle_number: int,
    later_cycle_number: int,
) -> None:
    tolerances = {
        "allocated": CUDA_ALLOCATED_GROWTH_TOLERANCE,
        "reserved": CUDA_RESERVED_GROWTH_TOLERANCE,
        "rss": RSS_GROWTH_TOLERANCE,
    }
    excessive_growth = []
    deltas = []
    for metric, tolerance in tolerances.items():
        delta = later_cycle[metric] - earlier_cycle[metric]
        deltas.append(f"{metric}={delta / MIB:+.0f}MiB")
        if delta > tolerance:
            excessive_growth.append(
                f"{metric} grew by {delta / MIB:.0f} MiB "
                f"(allowance {tolerance / MIB:.0f} MiB)"
            )
    print(
        f"{phase} cycle {later_cycle_number} - cycle {earlier_cycle_number}: "
        f"{', '.join(deltas)}"
    )
    if excessive_growth:
        raise AssertionError(
            f"Memory accumulated at the {phase} phase across switch cycles: "
            + "; ".join(excessive_growth)
        )


def log_memory_delta(
    phase: str,
    earlier_cycle: dict[str, int],
    later_cycle: dict[str, int],
    earlier_cycle_number: int,
    later_cycle_number: int,
) -> None:
    deltas = [
        f"{metric}={(later_cycle[metric] - earlier_cycle[metric]) / MIB:+.0f}MiB"
        for metric in ("allocated", "reserved", "rss")
    ]
    print(
        f"{phase} cycle {later_cycle_number} - cycle {earlier_cycle_number} "
        f"(informational): {', '.join(deltas)}"
    )


def assert_variant_downloaded(model_name: str, args: argparse.Namespace) -> None:
    from modern_iopaint.download import is_manifest_model_downloaded

    if not is_manifest_model_downloaded(
        model_name,
        precision=args.precision,
        rank=args.rank,
        lightning_steps=args.steps,
    ):
        raise RuntimeError(
            f"{model_name} ({args.precision}/{args.rank}/lightning-{args.steps}) "
            "is not fully cached. Download it with the matching P2 options first."
        )


def stage_manifest() -> None:
    from modern_iopaint.model_manifest import load_model_manifest

    manifest = load_model_manifest()
    qwen = manifest.get("qwen-image")
    edit = manifest.get("qwen-image-edit")
    if manifest.models["qwen-image-edit-2509"].integrated:
        raise AssertionError("2509 placeholder must not be integrated in P2")
    for record in (qwen, edit):
        if "transformer/*" not in record.base.ignore_patterns:
            raise AssertionError(f"{record.name} base snapshot does not exclude transformer/*")
        record.filename("fp4", "r32", 8)
        record.filename("int4", "r128", 4)
        record.filename("fp4", "r32", 0)
    print(f"manifest_version={manifest.version}, records={len(manifest.models)}")


def run_qwen_inference(
    model_name: str,
    args: argparse.Namespace,
    image,
    mask,
    *,
    edit: bool,
) -> None:
    import numpy as np

    torch = require_cuda()
    from modern_iopaint.model_manager import ModelManager

    assert_variant_downloaded(model_name, args)
    manager = ModelManager(
        name=model_name,
        device=torch.device("cuda"),
        **manager_kwargs(args),
    )
    try:
        if manager.model.lightning_steps != args.steps:
            raise AssertionError(
                f"Loaded lightning-{manager.model.lightning_steps}, expected {args.steps}"
            )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = manager(image, mask, make_request(args.steps, edit=edit))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(
            f"{model_name}: elapsed={elapsed:.2f}s, "
            f"max_allocated={allocated / GIB:.2f}GiB, "
            f"max_reserved={reserved / GIB:.2f}GiB"
        )
        if output.shape != (1024, 1024, 3):
            raise AssertionError(f"Unexpected output shape: {output.shape}")
        if output.dtype != np.uint8:
            raise AssertionError(f"Unexpected output dtype: {output.dtype}")
    finally:
        manager.unload()


def stage_switch(args: argparse.Namespace) -> None:
    torch = require_cuda()
    from modern_iopaint.download import scan_models
    from modern_iopaint.model_manager import ModelManager

    assert_variant_downloaded("qwen-image", args)
    available = {
        model.name
        for model in scan_models(
            qwen_precision=args.precision,
            qwen_rank=args.rank,
            qwen_lightning_steps=args.steps,
        )
    }
    if "lama" not in available:
        raise RuntimeError("LaMa must be downloaded before running switch stage (d)")

    manager = ModelManager(
        name="qwen-image",
        device=torch.device("cuda"),
        **manager_kwargs(args),
    )
    try:
        qwen_snapshots = []
        lama_snapshots = []
        for cycle in (1, 2, 3):
            if manager.name != "qwen-image":
                manager.switch("qwen-image")
            qwen_snapshots.append(memory_snapshot(torch, f"cycle {cycle} qwen"))

            manager.switch("lama")
            lama_snapshots.append(memory_snapshot(torch, f"cycle {cycle} lama"))

        log_memory_delta("qwen", qwen_snapshots[0], qwen_snapshots[1], 1, 2)
        log_memory_delta("lama", lama_snapshots[0], lama_snapshots[1], 1, 2)
        assert_memory_stable("qwen", qwen_snapshots[1], qwen_snapshots[2], 2, 3)
        assert_memory_stable("lama", lama_snapshots[1], lama_snapshots[2], 2, 3)
    finally:
        manager.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=("auto", "int4", "fp4"),
        default="auto",
    )
    parser.add_argument("--rank", choices=("r32", "r128"), default="r32")
    parser.add_argument(
        "--steps",
        choices=(4, 8),
        default=8,
        type=int,
        help="select the lightning variant and inference step count",
    )
    parser.add_argument("--skip-edit", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("auto", "fast", "balanced", "conservative"),
        default="auto",
        help="override automatic runtime profile selection",
    )
    args = parser.parse_args()

    inputs: dict[str, Any] = {}

    def qwen_stage() -> None:
        inputs["image"], inputs["mask"] = make_test_inputs()
        run_qwen_inference(
            "qwen-image", args, inputs["image"], inputs["mask"], edit=False
        )

    def edit_stage() -> None:
        if args.skip_edit:
            print("SKIP qwen-image-edit (--skip-edit)")
            return
        if "image" not in inputs:
            inputs["image"], inputs["mask"] = make_test_inputs()
        run_qwen_inference(
            "qwen-image-edit", args, inputs["image"], inputs["mask"], edit=True
        )

    results = [
        run_stage("(a) manifest", stage_manifest),
        run_stage("(b) qwen-image-inpaint", qwen_stage),
        run_stage("(c) qwen-image-edit-inpaint", edit_stage),
        run_stage("(d) model-switch-residency", lambda: stage_switch(args)),
    ]
    print("PASS smoke-p2" if all(results) else "FAIL smoke-p2")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
