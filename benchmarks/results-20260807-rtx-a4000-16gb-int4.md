# Modern-IOPaint VRAM benchmark — 20260807-023141+0000

> **Screening warning:** This is a screening/estimation tool only, **NOT certification of real N-GiB hardware behavior**. A PyTorch process cap does not emulate Windows display-memory reservation, allocator fragmentation, or non-PyTorch VRAM consumers.

- Preset: `quick`
- GPU: `NVIDIA RTX A4000`
- NVIDIA driver: `550.127.08`
- Total VRAM: `15.72 GiB`
- Total system RAM: `503.53 GiB`
- Nunchaku precision mode: `int4`
- Per-process VRAM cap: `none`
- Dependency tuple: `python 3.12.13`, `torch 2.11.0+cu128`, `diffusers 0.36.0`, `transformers 4.57.6`, `nunchaku 1.2.1+cu12.8torch2.11`

Load time is measured separately from inference and repeated across that model/profile's resolution rows. The first generation in each cell is a warmup; steady-state wall time is the second generation. The first model in a profile uses `ModelManager` initialization, subsequent models use `ModelManager.switch()`, and profile boundaries use `ModelManager.unload()`.

| Model | Resolution | Profile | Status / skip reason | Load time (s) | Steady-state wall (s) | Max allocated | Max reserved | Driver free/total before | Driver free/total after | Peak RSS | Steps/settings |
|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---|
| qwen-image | 1024×1024 | fast | ERROR: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 15.72 GiB of which 913.44 MiB is free. Process 772149 has 14.83 GiB memory in use. Of the allocated memory 14.48 GiB is allocated by PyTorch,... | 18.338 | — | 14.48 GiB | 14.67 GiB | 15.56 GiB / 15.72 GiB | 0.89 GiB / 15.72 GiB | 16.25 GiB | steps=8; guidance=1; strength=1; HDStrategy=Original; scheduler=FlowMatchEulerDiscreteScheduler; int4/r32/lightning-8 |
| lama | 1024×1024 | fast | OK | 1.167 | 2.559 | 0.97 GiB | 1.94 GiB | 15.36 GiB / 15.72 GiB | 13.58 GiB / 15.72 GiB | 11.65 GiB | steps=n/a (feed-forward); HDStrategy=Original; runtime profile=fast (not used by backend) |
| qwen-image | 1024×1024 | conservative | OK | 4.761 | 22.341 | 2.95 GiB | 2.98 GiB | 13.60 GiB / 15.72 GiB | 13.52 GiB / 15.72 GiB | 26.99 GiB | steps=8; guidance=1; strength=1; HDStrategy=Original; scheduler=FlowMatchEulerDiscreteScheduler; int4/r32/lightning-8; GPU blocks=8 |
| lama | 1024×1024 | conservative | OK | 1.781 | 2.516 | 0.99 GiB | 1.96 GiB | 15.26 GiB / 15.72 GiB | 13.52 GiB / 15.72 GiB | 12.53 GiB | steps=n/a (feed-forward); HDStrategy=Original; runtime profile=conservative (not used by backend) |
