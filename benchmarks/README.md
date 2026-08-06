# VRAM benchmark results

`scripts/bench_vram.py` is Modern-IOPaint's offline VRAM screening harness. It
measures model load/switch time separately, uses a first generation as warmup,
times the second generation, and records PyTorch peak allocated/reserved CUDA
memory, driver free/total VRAM before and after each cell, and sampled process
peak RSS. The 2048px cells use the application's mask-crop paths rather than
full-resolution diffusion inference.

The harness never downloads weights. Missing local models and unavailable
Nunchaku backends are written as `SKIP` rows.

Run the short iteration preset:

```text
python scripts/bench_vram.py --quick
```

Run the full matrix, optionally adding a local SDXL checkpoint:

```text
python scripts/bench_vram.py --full
python scripts/bench_vram.py --full --checkpoint C:\path\to\model.safetensors
python scripts/bench_vram.py --full --checkpoint C:\path\to\model.safetensors --cap-gib 12 --precision int4
```

Run a capped screening pass:

```text
python scripts/bench_vram.py --quick --cap-gib 12 --precision int4
```

The cap is only an estimate; it is not equivalent to physical 12 GiB hardware.
Official/published project measurements live in this directory as
`results-*.md` files.
