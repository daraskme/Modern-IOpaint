"""Platform-aware installer for the CUDA 12.8 Nunchaku runtime."""

from __future__ import annotations

import errno
import hashlib
import importlib.resources
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


CUDA_VERSION = (12, 8)
MIN_FREE_BYTES = 10 * 1024**3
TORCH_REQUIREMENTS = ("torch~=2.11.0", "torchvision~=0.26.0")
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


class GPUSetupError(RuntimeError):
    """An actionable setup error already formatted for the CLI."""


@dataclass(frozen=True)
class GPUInfo:
    name: str
    driver_version: str
    compute_capability: str | None


@dataclass(frozen=True)
class Installer:
    argv: tuple[str, ...]
    is_uv: bool
    label: str


@dataclass(frozen=True)
class TorchStackProbe:
    torch_version: str | None
    torchvision_version: str | None
    nunchaku_version: str | None
    torch_cuda_version: str | None
    nms_works: bool
    detail: str

    @property
    def is_cuda_build(self) -> bool:
        return self.torch_cuda_version is not None


def _run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise GPUSetupError(
            "ERROR [NO_NVIDIA_GPU]: nvidia-smi could not be started. Install an "
            "NVIDIA driver and confirm nvidia-smi is available on PATH."
        ) from error


def _detect_gpus(nvidia_smi: str) -> list[GPUInfo]:
    query_with_capability = [
        nvidia_smi,
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = _run_capture(query_with_capability)
    has_compute_capability = result.returncode == 0
    if not has_compute_capability:
        result = _run_capture(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "nvidia-smi returned no GPU records"
        raise GPUSetupError(
            "ERROR [NO_NVIDIA_GPU]: No NVIDIA GPU was found. "
            f"nvidia-smi diagnostic: {detail}"
        )

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        capability = fields[2] if has_compute_capability and len(fields) >= 3 else None
        if capability in ("", "N/A", "[N/A]"):
            capability = None
        gpus.append(GPUInfo(fields[0], fields[1], capability))
    if not gpus:
        raise GPUSetupError(
            "ERROR [NO_NVIDIA_GPU]: nvidia-smi ran, but no usable NVIDIA GPU "
            "record could be parsed."
        )
    return gpus


def _driver_cuda_version(nvidia_smi: str) -> tuple[int, int] | None:
    result = _run_capture([nvidia_smi])
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _platform_tag() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows" and machine in ("amd64", "x86_64"):
        return "win_amd64"
    if system == "Linux" and machine in ("amd64", "x86_64"):
        return "linux_x86_64"
    raise GPUSetupError(
        "ERROR [NO_MATCHING_WHEEL]: Nunchaku 1.2.1 CUDA 12.8 wheels are "
        f"not provided for {system} {platform.machine()}."
    )


def _python_tag() -> str:
    version = sys.version_info[:2]
    if version < (3, 10) or version > (3, 13):
        raise GPUSetupError(
            "ERROR [UNSUPPORTED_PYTHON]: setup-gpu supports Python 3.10 through "
            f"3.13; this interpreter is Python {version[0]}.{version[1]}."
        )
    return f"cp{version[0]}{version[1]}"


def _load_manifest() -> dict:
    resource = importlib.resources.files("modern_iopaint").joinpath("gpu_wheels.json")
    try:
        return json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GPUSetupError(
            f"ERROR [NO_MATCHING_WHEEL]: Cannot read bundled gpu_wheels.json: {error}"
        ) from error


def _select_wheel(manifest: dict, python_tag: str, platform_tag: str) -> dict:
    for wheel in manifest.get("wheels", []):
        if (
            wheel.get("python_tag") == python_tag
            and wheel.get("platform_tag") == platform_tag
        ):
            return wheel
    raise GPUSetupError(
        "ERROR [NO_MATCHING_WHEEL]: The bundled manifest has no Nunchaku "
        f"wheel for {python_tag}/{platform_tag}/CUDA 12.8/torch 2.11."
    )


def _choose_installer() -> Installer:
    try:
        uv_importable = importlib.util.find_spec("uv") is not None
    except (ImportError, AttributeError, ValueError):
        uv_importable = False
    if uv_importable:
        return Installer(
            (
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
            ),
            True,
            "uv",
        )
    uv_executable = shutil.which("uv")
    if uv_executable:
        return Installer(
            (uv_executable, "pip", "install", "--python", sys.executable),
            True,
            "uv",
        )
    return Installer((sys.executable, "-m", "pip", "install"), False, "pip")


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _run_install(
    installer: Installer,
    arguments: Sequence[str],
    description: str,
    dry_run: bool,
    echo: Callable[[str], None],
) -> None:
    command = [*installer.argv, *arguments]
    if dry_run:
        echo(f"DRY RUN: {_format_command(command)}")
        return
    echo(description)
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise GPUSetupError(
            f"ERROR [INSTALL_FAILED]: Could not start {installer.label}: {error}"
        ) from error
    if result.returncode != 0:
        raise GPUSetupError(
            f"ERROR [INSTALL_FAILED]: {description} failed with exit code "
            f"{result.returncode}."
        )


def _probe_torch_stack() -> TorchStackProbe:
    check = r"""
import importlib.metadata
import json


def installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


payload = {
    "torch_version": installed_version("torch"),
    "torchvision_version": installed_version("torchvision"),
    "nunchaku_version": installed_version("nunchaku"),
    "torch_cuda_version": None,
    "nms_works": False,
    "problems": [],
}
try:
    import torch

    payload["torch_version"] = str(torch.__version__)
    if torch.version.cuda is not None:
        payload["torch_cuda_version"] = str(torch.version.cuda)
    else:
        payload["problems"].append("torch.version.cuda is None")
except Exception as error:
    payload["problems"].append(
        f"torch import failed: {type(error).__name__}: {error}"
    )
else:
    try:
        import torchvision

        payload["torchvision_version"] = str(torchvision.__version__)
        nms = getattr(torchvision.ops, "nms", None)
        if not callable(nms):
            raise RuntimeError("torchvision.ops.nms is missing or not callable")
        nms(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            0.5,
        )
        payload["nms_works"] = True
    except Exception as error:
        payload["problems"].append(
            f"torchvision.ops.nms verification failed: "
            f"{type(error).__name__}: {error}"
        )

print("MODERN_IOPAINT_TORCH_PROBE=" + json.dumps(payload, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", check],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return TorchStackProbe(
            None,
            None,
            None,
            None,
            False,
            f"torch probe could not start: {error}",
        )

    prefix = "MODERN_IOPAINT_TORCH_PROBE="
    payload: dict | None = None
    for line in reversed(result.stdout.splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            candidate = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break

    if payload is None:
        stderr = result.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "torch probe returned no data"
        return TorchStackProbe(None, None, None, None, False, detail)

    problems = payload.get("problems")
    if isinstance(problems, list):
        detail = "; ".join(str(problem) for problem in problems if problem)
    else:
        detail = ""
    return TorchStackProbe(
        payload.get("torch_version"),
        payload.get("torchvision_version"),
        payload.get("nunchaku_version"),
        payload.get("torch_cuda_version"),
        payload.get("nms_works") is True,
        detail or "probe completed successfully",
    )


def _format_version_diagnostics(probe: TorchStackProbe) -> str:
    return (
        f"Versions: torch={probe.torch_version or 'not installed/unknown'}, "
        f"torchvision={probe.torchvision_version or 'not installed/unknown'}, "
        f"nunchaku={probe.nunchaku_version or 'not installed/unknown'}"
    )


def _verify_torch_stack(echo: Callable[[str], None]) -> TorchStackProbe:
    probe = _probe_torch_stack()
    versions = _format_version_diagnostics(probe)
    if not probe.is_cuda_build or not probe.nms_works:
        raise GPUSetupError(
            "ERROR [TORCH_CUDA_SWAP_FAILED]: The torch/torchvision CUDA 12.8 "
            "swap did not take effect or produced an inconsistent stack. Expected "
            "torch.version.cuda to be non-None and torchvision.ops.nms to import "
            f"and execute successfully. {versions}. Probe diagnostic: {probe.detail}"
        )
    echo(
        f"CUDA stack verification: {versions}; "
        f"torch.version.cuda={probe.torch_cuda_version}; "
        "torchvision.ops.nms=working"
    )
    return probe


def _ensure_free_disk(path: Path) -> None:
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as error:
        raise GPUSetupError(
            f"ERROR [INSUFFICIENT_DISK]: Free disk space could not be checked: {error}"
        ) from error
    if free_bytes < MIN_FREE_BYTES:
        free_gib = free_bytes / 1024**3
        raise GPUSetupError(
            "ERROR [INSUFFICIENT_DISK]: setup-gpu requires at least 10 GiB of "
            f"working space; only {free_gib:.1f} GiB is free on {path.anchor or path}."
        )


def _download_wheel(wheel: dict, destination: Path) -> None:
    url = str(wheel.get("url", ""))
    expected_hash = str(wheel.get("sha256", "")).strip().lower()
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > shutil.disk_usage(destination.parent).free:
                raise GPUSetupError(
                    "ERROR [INSUFFICIENT_DISK]: There is not enough temporary disk "
                    "space for the selected Nunchaku wheel."
                )
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
    except GPUSetupError:
        raise
    except OSError as error:
        if getattr(error, "errno", None) == errno.ENOSPC:
            raise GPUSetupError(
                "ERROR [INSUFFICIENT_DISK]: Disk space ran out while downloading "
                "the Nunchaku wheel."
            ) from error
        raise GPUSetupError(
            f"ERROR [DOWNLOAD_FAILED]: Could not download {url}: {error}"
        ) from error
    except (urllib.error.URLError, ValueError) as error:
        raise GPUSetupError(
            f"ERROR [DOWNLOAD_FAILED]: Could not download {url}: {error}"
        ) from error

    actual_hash = digest.hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise GPUSetupError(
            "ERROR [DOWNLOAD_FAILED]: SHA-256 mismatch for the Nunchaku wheel; "
            f"expected {expected_hash}, downloaded {actual_hash}."
        )


def _verify_nunchaku(echo: Callable[[str], None]) -> None:
    verification = r"""
import importlib.metadata


def installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed/unknown"


print(
    "Versions: "
    f"torch={installed_version('torch')}, "
    f"torchvision={installed_version('torchvision')}, "
    f"nunchaku={installed_version('nunchaku')}",
    flush=True,
)
import nunchaku
from nunchaku.utils import get_precision

print("get_precision()=" + str(get_precision()))
"""
    result = subprocess.run(
        [sys.executable, "-c", verification],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stdout_lines = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        version_detail = next(
            (line for line in stdout_lines if line.startswith("Versions: ")),
            "Versions: unavailable",
        )
        non_version_stdout = "\n".join(
            line for line in stdout_lines if line != version_detail
        )
        detail = result.stderr.strip() or non_version_stdout or "no error output"
        raise GPUSetupError(
            "ERROR [VERIFY_FAILED]: Nunchaku import/get_precision() verification "
            f"failed. {version_detail}. Diagnostic: {detail}"
        )
    for line in result.stdout.splitlines():
        if line.strip():
            echo(f"Verification: {line.strip()}")


def run_setup_gpu(
    *, dry_run: bool = False, echo: Callable[[str], None] = print
) -> None:
    """Install and verify the supported CUDA/Nunchaku tuple."""

    python_tag = _python_tag()
    platform_tag = _platform_tag()
    echo(
        f"Host: {platform.system()} {platform.machine()}, "
        f"Python {platform.python_version()} ({python_tag})"
    )
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise GPUSetupError(
            "ERROR [NO_NVIDIA_GPU]: nvidia-smi was not found. Install an NVIDIA "
            "driver and confirm an NVIDIA GPU is visible before retrying."
        )

    gpus = _detect_gpus(nvidia_smi)
    for index, gpu in enumerate(gpus):
        suffix = (
            f", compute capability {gpu.compute_capability}"
            if gpu.compute_capability
            else ", compute capability unavailable"
        )
        echo(f"GPU {index}: {gpu.name}, driver {gpu.driver_version}{suffix}")

    driver_cuda = _driver_cuda_version(nvidia_smi)
    if driver_cuda is not None and driver_cuda < CUDA_VERSION:
        raise GPUSetupError(
            "ERROR [DRIVER_TOO_OLD]: The NVIDIA driver reports CUDA "
            f"{driver_cuda[0]}.{driver_cuda[1]} support, but CUDA 12.8 is required. "
            "Update the NVIDIA driver and retry."
        )
    if driver_cuda is None:
        echo(
            "WARNING: nvidia-smi did not report its supported CUDA version; "
            "continuing because the GPU query succeeded."
        )
    else:
        echo(f"Driver CUDA support: {driver_cuda[0]}.{driver_cuda[1]}")

    manifest = _load_manifest()
    wheel = _select_wheel(manifest, python_tag, platform_tag)
    echo(
        f"Selected {wheel['filename']} for Python {sys.version_info.major}."
        f"{sys.version_info.minor} on {platform_tag}."
    )
    if not str(wheel.get("sha256", "")).strip():
        echo(
            "WARNING: The selected manifest SHA-256 is empty. Release maintainers "
            "should run scripts/update_wheel_hashes.py before publishing."
        )

    temp_root = Path(tempfile.gettempdir()).resolve()
    _ensure_free_disk(temp_root)
    installer = _choose_installer()
    echo(f"Package installer: {installer.label}")

    torch_probe = _probe_torch_stack()
    cuda_status = (
        f"CUDA build (torch.version.cuda={torch_probe.torch_cuda_version})"
        if torch_probe.is_cuda_build
        else "non-CUDA build (torch.version.cuda=None)"
    )
    echo(
        "Current torch probe: "
        f"torch {torch_probe.torch_version or 'not installed/unknown'}; "
        f"{cuda_status}; {torch_probe.detail}"
    )
    if not torch_probe.is_cuda_build:
        torch_arguments: list[str] = []
        if installer.is_uv:
            torch_arguments.extend(
                (
                    "--torch-backend=cu128",
                    "--reinstall-package",
                    "torch",
                    "--reinstall-package",
                    "torchvision",
                )
            )
        else:
            # This dedicated invocation scopes --force-reinstall to torch and
            # torchvision while still allowing their dependencies to resolve.
            torch_arguments.extend(
                ("--force-reinstall", "--index-url", PYTORCH_CU128_INDEX)
            )
        torch_arguments.extend(TORCH_REQUIREMENTS)
        _run_install(
            installer,
            torch_arguments,
            "Installing torch 2.11 and torchvision 0.26 with CUDA 12.8 support...",
            dry_run,
            echo,
        )
    else:
        echo("CUDA-enabled torch is already available; leaving it in place.")

    if dry_run:
        echo(
            "DRY RUN: verify torch.version.cuda is non-None and execute "
            "torchvision.ops.nms before installing Nunchaku"
        )
    else:
        _verify_torch_stack(echo)

    if dry_run:
        echo(f"DRY RUN: download {wheel['url']}")
        preview_path = Path(wheel["filename"])
        _run_install(
            installer,
            ("--no-deps", "--force-reinstall", str(preview_path)),
            "Installing Nunchaku...",
            True,
            echo,
        )
        echo("DRY RUN: verify `import nunchaku` and `get_precision()` in a subprocess")
        return

    _ensure_free_disk(temp_root)
    with tempfile.TemporaryDirectory(prefix="modern-iopaint-gpu-") as temp_dir:
        wheel_path = Path(temp_dir) / str(wheel["filename"])
        echo(f"Downloading Nunchaku wheel from {wheel['url']}")
        _download_wheel(wheel, wheel_path)
        if wheel.get("sha256"):
            echo("Nunchaku wheel SHA-256 verified.")
        _run_install(
            installer,
            ("--no-deps", "--force-reinstall", str(wheel_path)),
            "Installing Nunchaku with --no-deps (preserving CUDA torch)...",
            False,
            echo,
        )

    _verify_nunchaku(echo)
    echo("GPU setup complete.")
