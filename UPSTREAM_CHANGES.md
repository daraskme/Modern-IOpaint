# Upstream changes

This file records modernization changes relative to IOPaint at upstream commit
`61a759fb3f332bacdce8b2813f4837495c9b86e0`.

## Phase P1: dependency modernization and tuple pinning

The primary target is Python 3.12, while project metadata allows Python
`>=3.10,<3.14`. The selected compatibility tuple is:

- `torch~=2.11.0`
- `diffusers==0.36.0`
- `transformers==5.14.1`
- `accelerate==1.14.0`
- `huggingface_hub==1.26.0`
- `peft==0.20.0`
- `fastapi>=0.141.1,<1`
- `pydantic>=2.13.4,<3`
- `python-socketio>=5.16.3,<6`
- `typer>=0.27.1,<1`
- `Pillow>=10`

`diffusers==0.36.0` is a required compatibility pin for Nunchaku 1.2.1. Its
Qwen transformer calls `QwenEmbedRope` with the 0.36 signature
`pos_embed(img_shapes, txt_seq_lens, device=...)`. Diffusers 0.39 changes that
signature and fails with a duplicate `device` argument; 0.37 and 0.38 pass no
usable text sequence length and fail validation. Nunchaku's own CI also pins
Diffusers 0.36. `transformers==5.14.1` is retained from the researched P1 tuple.
Gradio and `controlnet-aux` were removed from core dependencies.

### Quarantined features

The following source is retained for provenance under
`modern_iopaint/model/legacy/`, but is excluded from the model registry, model
discovery, API capability lists, and CLI choices:

- **AnyText**: its vendored latent-diffusion/OCR stack depends on old Diffusers
  and Transformers internals and was the reason for the upstream Pillow 9.5 pin.
- **BrushNet**: its vendored UNet, attention-processor, and LoRA integration uses
  Diffusers internals that are not compatible with the active dependency tuple.
- **PowerPaint v1 and v2**: their vendored pipelines depend on the old BrushNet,
  UNet, callback, and Diffusers internal APIs.

The Gradio `start-web-config` command is removed from the CLI. Its implementation
is retained as `modern_iopaint/legacy/web_config.py`, and the Windows helper is
retained as `scripts/legacy/win_config.bat`. Gradio 4.21 is incompatible with the
P1 FastAPI/Pydantic tuple and is no longer a core dependency.

OpenPose and depth ControlNet preprocessors were also quarantined by removing
`controlnet-aux` and their model choices. Plain Diffusers 0.36 ControlNet remains
enabled for Canny and inpaint control images, which are generated with OpenCV and
NumPy only.

### Retained compatibility changes

- Hugging Face model downloads use `snapshot_download`; `hf_hub_download` calls
  use explicit `repo_id` and `filename` keyword arguments.
- Local SD/SDXL checkpoint loading supplies the bundled YAML path through
  `original_config` rather than the deprecated `original_config_file` argument.
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
  (and the advertised Python 3.10-3.13 range), especially Diffusers 0.36 with
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

## Phase P2: Qwen backends and model-management foundation

P2 adds optional Nunchaku 1.2.1 backends for Qwen Image mask inpainting and
Qwen Image Edit instruction-based mask inpainting. Nunchaku is deliberately not
added to the core dependency tuple; users install its platform-appropriate
package separately. If it cannot be imported, both Qwen entries are omitted
from model discovery with an actionable warning, while LaMa, SD 1.5, and SDXL
continue to initialize normally.

### Versioned manifest and downloads

`modern_iopaint/model_manifest.json` is the source of truth for the two P2
models. It records the `nunchaku-ai` repositories, parameterized int4/fp4,
r32/r128, base/lightning filenames, licenses, gating, approximate sizes,
revisions, hashes, and snapshot filters. Qwen base components are fetched from
`Qwen/Qwen-Image` or `Qwen/Qwen-Image-Edit`; both base snapshots explicitly
exclude `transformer` and `transformer/*`, because Nunchaku's quantized
transformer replaces the approximately 41 GB bf16 transformer completely.

The Qwen Image Edit 2509 repositories and lightning v2 filename shapes are
represented by a non-integrated manifest placeholder. P2 does not register,
download, or load 2509.

Manifest downloads use `huggingface_hub.snapshot_download` with the declared
allow/ignore patterns, preserve the existing `--model-dir` and
`XDG_CACHE_HOME` behavior (plus standard `HF_HOME`/`HF_HUB_CACHE` overrides),
check free disk capacity before starting, and log each snapshot phase. Current
revisions are temporarily `main`; maintainers can run
`scripts/update_manifest_hashes.py` later to query Hub metadata, pin exact
commit revisions, and populate available LFS SHA-256 values.

### Runtime and request integration

`QwenImage` uses Diffusers 0.36 `QwenImageInpaintPipeline`, and `QwenImageEdit`
uses `QwenImageEditInpaintPipeline`. Both use
`NunchakuQwenImageTransformer2DModel`, bfloat16 pipeline components, the shared
diffusion crop/extend/mask-blur path, request seed/strength/steps/guidance, and
the existing model registry. They reuse `ModelType.DIFFUSERS_OTHER`, matching
the Paint-by-Example and InstructPix2Pix plumbing, with explicit capability
metadata so the frontend exposes prompt, steps, guidance, strength, crop, and
extend controls without a new model category.

The default selection is auto precision, r32, lightning 8-step, and automatic
runtime profiling. Lightning 4-step and non-lightning variants are selectable.
Lightning inference uses `true_cfg_scale=1.0` and does not send a negative
prompt. The frontend adopts per-model default steps/guidance when the current
model changes.

At model-load time, the runtime profiler records free/total VRAM and total
system RAM, then chooses:

- `fast` at 20 GiB or more free VRAM: model CPU offload;
- `balanced` from 13 GiB up to 20 GiB: model CPU offload plus VAE tiling;
- `conservative` below 13 GiB: computed Nunchaku transformer block residency,
  sequential CPU offload, and VAE tiling.

For the conservative profile, Nunchaku block offload is enabled first, the
pipeline's exact `transformer` component name is then added to
`_exclude_from_cpu_offload`, and only then is Diffusers sequential CPU offload
enabled for the remaining components. The exclusion is assigned as an instance
list even when Diffusers starts with an empty class-level list; otherwise an
unattached temporary list leaves the transformer exposed to Accelerate meta
hooks and produces `Cannot copy out of meta tensor` on its first forward pass.
Nunchaku's transformer `.to(...)` override ignores device moves while its own
offload is active, so the Diffusers exclusion path does not manually relocate
the quantized blocks.

`--runtime-profile` or `MODERN_IOPAINT_RUNTIME_PROFILE` can override automatic
selection. Qwen precision, rank, and lightning selection likewise have CLI and
environment overrides documented by command help.

### Large-model residency and verification

The model-manager switch path now removes Accelerate hooks, drops pipeline
references, collects Python objects, empties the CUDA cache, and logs recovered
free VRAM when entering or leaving Qwen/SD/SDXL backends. This enforces one
large diffusion pipeline resident at a time, including Qwen-to-SD-to-Qwen and
Qwen-to-LaMa-to-Qwen transitions.

The P2 switch smoke test is offload-aware: it performs
Qwen-to-LaMa-to-Qwen-to-LaMa switching, records CUDA allocated/reserved memory
and process RSS at both model phases in each cycle, and compares cycle two with
cycle one using fixed small growth tolerances. It retains teardown logging but
does not require free VRAM to rise when the offloaded Qwen weights were already
resident in system RAM.

On Windows, the download manager sets `HF_HUB_DISABLE_SYMLINKS=1` with
`os.environ.setdefault` before importing or calling Hugging Face Hub download
functions. This avoids `WinError 1314` when a cached blob is shared across
repositories on systems without Developer Mode, while preserving an explicit
user-provided setting.

Nunchaku transformer files may contain the cosmetic config key
`pooled_projection_dim=768`. Diffusers 0.36's Qwen transformer constructor does
not consume it, so Nunchaku logs that the attribute is unexpected. Nunchaku
1.2.1 reads the config directly from safetensors metadata and exposes no clean
caller-side filter, so the harmless message is left as a known cosmetic warning.

External P2 verification confirmed the manifest and both 1024x1024 Qwen
inpainting pipelines on the fast profile with Diffusers 0.36 and Nunchaku
1.2.1. The conservative exclusion fix and revised switch-cycle leak assertions
were made under a static-only constraint and still require an external rerun.
Run `python scripts/smoke_p2.py` after installing Nunchaku and downloading the
selected Qwen Image, Qwen Image Edit, and LaMa models. It validates the
manifest, performs masked inference for both Qwen pipelines, prints
elapsed/peak VRAM statistics, and checks two Qwen-to-LaMa switch cycles. Use
`--skip-edit` when only the non-edit model is installed.

## Phase P3: optional FLUX.1-Fill backend and license compliance

P3 adds `flux.1-fill-dev` as an optional `DIFFUSERS_OTHER` backend. It is only
registered and discovered when the platform-specific Nunchaku package imports
successfully. The backend inherits the existing diffusion crop, extend,
strength, mask-blur, deterministic seed, and unmasked-area compositing paths.
Its model defaults are 28 steps and guidance scale 30, and it deliberately does
not pass a negative prompt to Diffusers.

The implementation was grounded against the installed Nunchaku 1.2.1 and
Diffusers 0.36.0 sources. The exported FLUX class is
`NunchakuFluxTransformer2dModel` (lowercase `d` in `2d`), whose
`from_pretrained(path, **kwargs)` reads the `offload` flag while constructing
the quantized module. It does not expose Qwen's later `set_offload` method.
The optional text encoder is `NunchakuT5EncoderModel`; its
`from_pretrained(path, **kwargs)` accepts `torch_dtype` and `device`. The
Diffusers `FluxFillPipeline` accepts the original image and white-repaint mask,
defaults guidance to 30, and exposes VAE tiling through
`enable_vae_tiling()`.

Runtime profiles apply those APIs as follows:

- `fast`: Diffusers model CPU offload;
- `balanced`: model CPU offload plus VAE tiling;
- `conservative`: `offload=True` while loading the Nunchaku FLUX transformer,
  Diffusers sequential CPU offload, VAE tiling, and the optional int4
  Nunchaku T5-XXL as `text_encoder_2` when cached.

### Upstream-only weights and gated access

No FLUX transformer, base-model, T5, tokenizer, VAE, or other weight file is
bundled, mirrored, or copied into this repository or its application package.
Downloads go directly into the user's Hugging Face cache from exactly these
upstream repositories:

- `nunchaku-ai/nunchaku-flux.1-fill-dev` for the selected r32 int4/fp4
  transformer;
- `black-forest-labs/FLUX.1-Fill-dev` for scheduler, tokenizers, CLIP,
  T5-XXL, and VAE components;
- `nunchaku-ai/nunchaku-t5` for the optional int4 T5-XXL.

The BFL snapshot allowlist contains only `model_index.json`, scheduler,
tokenizers, both text encoders, and VAE. Its ignore list independently excludes
`transformer`, `transformer/*`, and the duplicate root-level
`flux1-fill-dev.safetensors`, so neither full transformer representation is
downloaded from the gated base repository. The manifest records approximately
6.8 GiB for one quantized transformer, 10 GiB for the base components, and
2.5 GiB for the optional int4 T5, with a combined disk preflight estimate.

The BFL repository remains gated. A 401, 403, or `GatedRepoError` is converted
to a structured `FluxAccessError` that tells the user to log into Hugging Face,
accept the FLUX.1-dev license on the model page, and authenticate through
`huggingface-cli login`/`hf auth login` or `HF_TOKEN`. Offline empty-cache
access produces the same actionable result; gating is never bypassed.
`modern-iopaint check-flux-access` reports authentication and repository
license-access status using the installed Hugging Face Hub `whoami` and
`model_info` APIs.

### License notice and persisted acceptance

The manifest identifies the model as licensed under the
**FLUX.1-dev Non-Commercial License** and links to Black Forest Labs' full
license text. The model selector displays a non-commercial badge. On first
selection, the frontend requires explicit confirmation of a notice covering
non-commercial model use, the separate commercial-license requirement,
permitted output use, model restrictions, and content-filtering/output-review
obligations.

Acceptance is stored server-side in `modern_iopaint_settings.json` under the
configured model/cache root, not merely in browser storage. The model-switch
API refuses FLUX selection until that record exists, and subsequent frontend
sessions read the persisted acceptance so the notice is only shown once.

P3 was implemented statically without Python execution, network access, or
runtime model loading. Run `python scripts/smoke_p3.py` externally. It validates
the manifest, performs 1024x1024 eight-step masked inpainting under fast and
conservative profiles with elapsed/peak-VRAM reporting, and checks the
structured gated-error message. GPU/weight/auth-dependent stages print a clear
`SKIP` when their prerequisites are unavailable.

## Phase P4: illustration and anime support polish

P4 promotes the existing `anime-lama` TorchScript backend to first-class model
metadata. The manifest records its upstream-hosted source URL, approximate
download size, MIT license, `dreMaz/AnimeMangaInpainting` provenance, and
`erase-illustration` category. Model discovery now returns one of five explicit
categories for every entry: photo or illustration erase, general inpainting,
or photo or illustration inpainting. The frontend keeps the existing model-type
tabs and groups each list under the bilingual `実写/Photo`,
`イラスト/Illustration`, and `汎用/General` headings.

Local single-file SD and SDXL discovery now reads an optional same-name JSON
sidecar beside each `.ckpt` or `.safetensors` file. The sidecar supports
`prediction_type` and `category`; local checkpoints default to
`inpaint-photo`, while illustration checkpoints can opt into
`inpaint-illustration`. After Diffusers 0.36 constructs the pipeline, an explicit
`prediction_type` is registered on the scheduler config before inference. This
corrects unreliable v-prediction inference for community SDXL checkpoints and
is preserved when the application replaces the scheduler for the selected
sampler.

No third-party SDXL checkpoint is bundled or endorsed. Locally loaded model
licenses and permitted usage remain the user's responsibility. P4 was
implemented under the same static-only constraint: no Python, Git, network, or
runtime model execution was used. Run `python scripts/smoke_p4.py` externally;
use `--checkpoint PATH` for the two-step 512px local SDXL stage and
`--skip-qwen` when the P2 Qwen runtime is unavailable. The script always reports
a clear `PASS`, `FAIL`, or `SKIP` for its AnimeLaMa, local SDXL, and Qwen stages.

P4 runtime verification found that Transformers 5.x is incompatible with
Diffusers 0.36 single-file CLIP conversion during `from_single_file`.
Transformers is therefore pinned to `4.57.6`, which has been verified working
across the P1, P2, and P3 smoke suites.
