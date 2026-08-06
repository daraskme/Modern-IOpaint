# Upstream changes

This file records modernization changes relative to IOPaint at upstream commit
`61a759fb3f332bacdce8b2813f4837495c9b86e0`.

## Phase P1: dependency modernization and tuple pinning

The primary target is Python 3.12, while project metadata allows Python
`>=3.10,<3.14`. The selected compatibility tuple is:

- `torch~=2.11.0`
- `diffusers==0.39.0`
- `transformers==5.14.1`
- `accelerate==1.14.0`
- `huggingface_hub==1.26.0`
- `peft==0.20.0`
- `fastapi>=0.141.1,<1`
- `pydantic>=2.13.4,<3`
- `python-socketio>=5.16.3,<6`
- `typer>=0.27.1,<1`
- `Pillow>=10`

`transformers==5.14.1` is retained from the researched P1 tuple. The repository
contains no local dependency metadata proving that Diffusers 0.39 requires
Transformers 4.x; the external installation must confirm that combination.
Gradio and `controlnet-aux` were removed from core dependencies.

### Quarantined features

The following source is retained for provenance under
`modern_iopaint/model/legacy/`, but is excluded from the model registry, model
discovery, API capability lists, and CLI choices:

- **AnyText**: its vendored latent-diffusion/OCR stack depends on old Diffusers
  and Transformers internals and was the reason for the upstream Pillow 9.5 pin.
- **BrushNet**: its vendored UNet, attention-processor, and LoRA integration uses
  Diffusers internals that are not compatible with Diffusers 0.39.
- **PowerPaint v1 and v2**: their vendored pipelines depend on the old BrushNet,
  UNet, callback, and Diffusers internal APIs.

The Gradio `start-web-config` command is removed from the CLI. Its implementation
is retained as `modern_iopaint/legacy/web_config.py`, and the Windows helper is
retained as `scripts/legacy/win_config.bat`. Gradio 4.21 is incompatible with the
P1 FastAPI/Pydantic tuple and is no longer a core dependency.

OpenPose and depth ControlNet preprocessors were also quarantined by removing
`controlnet-aux` and their model choices. Plain Diffusers 0.39 ControlNet remains
enabled for Canny and inpaint control images, which are generated with OpenCV and
NumPy only.

### Retained compatibility changes

- Hugging Face model downloads use `snapshot_download`; `hf_hub_download` calls
  use explicit `repo_id` and `filename` keyword arguments.
- Local SD/SDXL checkpoint loading supplies parsed `original_config` data rather
  than the removed `original_config_file`/`load_safety_checker` arguments.
- Diffusers callbacks use `callback_on_step_end` and return `callback_kwargs`.
- The CPU text-encoder wrapper is a plain `torch.nn.Module` instead of depending
  on Transformers' `PreTrainedModel` call internals.
- Pydantic request validation uses a Pydantic 2 instance model validator.
- Pillow resampling uses `Image.Resampling.LANCZOS`.
- PEFT/LCM adapter detection tolerates both `get_list_adapters()` and the
  pipeline `peft_config` mapping.

The active registry continues to expose LaMa, AnimeLaMa, MIGAN, OpenCV inpaint,
standard SD 1.5 and SDXL inpaint pipelines, supported style checkpoints, local
single-file checkpoints, and the mask crop/extend mechanisms. This phase does
not add Nunchaku or Qwen support.

### Runtime verification required

P1 was implemented under a static-only constraint: no dependency installation,
Python execution, or network access was performed. External verification should
therefore cover:

- resolution and import compatibility of the complete tuple on Python 3.12
  (and the advertised Python 3.10-3.13 range), especially Diffusers 0.39 with
  Transformers 5.14.1 and torch 2.11;
- LaMa/AnimeLaMa/MIGAN TorchScript loading and CPU inference under torch 2.11;
- SD 1.5 and SDXL directory and `from_single_file` loading, safety-checker
  disabling, scheduler selection, and `callback_on_step_end` behavior;
- PEFT 0.20 LCM-LoRA load/enable/disable behavior;
- Canny and inpaint ControlNet loading, preprocessing, switching, and inference;
- FastAPI request parsing with Pydantic 2.13 and Typer/`typer-config` CLI startup;
- Hugging Face Hub 1.26 cache discovery and online/offline download behavior;
- optional Transformers-backed plugins and the unchanged frontend/server API
  contract.

Run `python scripts/smoke_p1.py` for the required CPU LaMa check, and add `--gpu`
to request the three-step 256x256 SD 1.5 CUDA check.
