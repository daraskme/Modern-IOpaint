> **Fork notice:** Modern-IOPaint is a modernization fork of [IOPaint](https://github.com/Sanster/IOPaint), based on upstream commit `61a759fb3f332bacdce8b2813f4837495c9b86e0`. It remains licensed under Apache-2.0; see [NOTICE](NOTICE) for attribution and modification tracking.

# Modern-IOPaint

## 日本語

Modern-IOPaint は、[Sanster/IOPaint](https://github.com/Sanster/IOPaint) を現行の
Python、PyTorch、Diffusers 環境向けに更新したセルフホスト型の画像インペイント／
アウトペイント・アプリケーションです。ブラウザー UI から不要物の消去、マスク領域の
生成・編集、イラスト修正などをローカル環境で実行できます。

### 主な機能

- `qwen-image` と `qwen-image-edit` によるインペイント／指示編集。Nunchaku 1.2.1
  の 4-bit 量子化（int4/fp4、対応 GPU に応じて自動選択）と 4/8-step Lightning
  アダプターに対応します。
- オプションの `flux.1-fill-dev` バックエンド。FLUX.1-dev Non-Commercial
  License が適用され、初回利用時にアプリ内でライセンス確認と同意が必要です。
- 写真向け消去モデルに加え、`anime-lama` と独立した Illustration カテゴリーを提供します。
- ローカルの SD/SDXL `.ckpt`／`.safetensors` を読み込み可能です。同名 JSON
  サイドカーで `prediction_type` と `category` を指定できます。
- モデルの切り替え、マスクの crop/extend、各種補助プラグイン、ファイル管理を
  既存 IOPaint の操作系から継承しています。

ローカルチェックポイントの例:

```json
{
  "prediction_type": "v_prediction",
  "category": "inpaint-illustration"
}
```

この JSON を `example.safetensors` と同じ場所に `example.json` として置きます。

### 推奨環境

- Windows x64 または Linux x86-64
- CUDA 12.8 をサポートする NVIDIA ドライバーと NVIDIA GPU
- 12 GB 以上の VRAM（推奨。大きな画像やモデルではさらに必要な場合があります）
- 32 GB 以上のシステム RAM
- 45 GB 以上の空きディスク容量
- Python 3.10～3.13（配布用・推奨検証環境は Python 3.12）

検証済み依存関係は torch `~=2.11.0` + cu128、torchvision `~=0.26.0`、
Diffusers `0.36.0`、Transformers `4.57.6`、Nunchaku `1.2.1` です。

### pip でインストール

新しい仮想環境で次を実行します。`setup-gpu` は GPU、ドライバー、Python を確認し、
CUDA 版 torch/torchvision と、プラットフォームに合う Nunchaku wheel を設定します。
Nunchaku は依存関係を解決させず `--no-deps` で導入されるため、CUDA 版 torch が
PyPI の CPU 版へ置き換えられる既知の問題を回避します。

```console
pip install modern-iopaint
modern-iopaint setup-gpu
modern-iopaint start --model qwen-image
```

既定の `--device auto` は CUDA が利用可能なら `cuda`、それ以外は `cpu` を選びます。
`--device cpu`、`--device cuda`、`--device mps` で明示的に上書きできます。

互換性と予定コマンドのみ確認する場合:

```console
modern-iopaint setup-gpu --dry-run
```

### Windows ワンクリック版

Release の `modern-iopaint-oneclick-<version>.zip` を書き込み可能なフォルダーへ展開し、
`run.bat` をダブルクリックします。これはオンライン・ブートストラップです。固定済み uv、
uv 管理の Python 3.12、アプリ、CUDA 12.8 依存関係を取得した後、
`qwen-image` をポート 8080 で起動します。インターネット接続が必要です。

アプリと依存関係のダウンロードは約 4 GB、モデルは選択内容により約 12～40 GB が
目安です。全処理は `setup.log` に記録されます。失敗時はコンソールが閉じないため、
メッセージを確認できます。再実行は既存環境を修復・更新します。

### モデルとライセンス

| モデル／重み | ライセンス | 注意事項 |
|---|---|---|
| Qwen Image／Qwen Image Edit、Nunchaku 量子化、Lightning | Apache-2.0 | 4-bit 推論用。重みは上流配布元から取得します。 |
| MIGAN | MIT | 消去モデル。上流のライセンスと配布条件を確認してください。 |
| AnimeLaMa | MIT | `dreMaz/AnimeMangaInpainting` 由来のイラスト向け消去モデルです。 |
| FLUX.1-Fill-dev | FLUX.1-dev Non-Commercial License | オプション。非商用条件が適用され、アプリ内同意が必要です。 |
| ユーザー提供ローカルチェックポイント | 提供者ごとに異なる | 提供・利用する人がライセンス、権利、許可された用途を確認する責任を負います。 |

このリポジトリと PyPI wheel はモデル重みを同梱しません。

### ベンチマーク

測定方法と既存レポートは [`benchmarks/`](benchmarks/) にあります。結果はハードウェア、
ドライバー、精度、runtime profile に強く依存します。正式なリリース比較表は実機測定後に
更新します。

| GPU | VRAM | モデル／設定 | 1024px 時間 | Peak VRAM | レポート |
|---|---:|---|---:|---:|---|
| TBD | TBD | Qwen Image int4, 8-step Lightning | TBD | TBD | [`benchmarks/README.md`](benchmarks/README.md) |
| TBD | TBD | FLUX Fill int4 | TBD | TBD | [`benchmarks/README.md`](benchmarks/README.md) |

### 開発

バックエンドとフロントエンドをそれぞれ準備します。wheel を作る前には必ずフロントエンドを
build してください。Hatch が `web_app/dist` を wheel 内の
`modern_iopaint/web_app` へ直接取り込むため、手動コピーは不要です。

```console
pip install -e .
cd web_app
npm ci
npm run build
cd ..
modern-iopaint start --model lama
```

開発ツリーでは package 内の assets がまだ無い場合、サーバーは `web_app/dist` を
検出してログ通知のうえ使用します。配布物を作るには `hatch build` を使います。
Nunchaku wheel の SHA-256 はリリース前に `python scripts/update_wheel_hashes.py`、
ワンクリック版の uv pin は `python scripts/update_oneclick_pins.py --version <tag>` で
更新してください。

### クレジット

本プロジェクトは Sanster と IOPaint contributors による
[IOPaint](https://github.com/Sanster/IOPaint) を基盤としています。コードは
[Apache License 2.0](LICENSE) の条件で提供されます。上流の帰属、基準 commit、変更の
概要は [NOTICE](NOTICE) と [UPSTREAM_CHANGES.md](UPSTREAM_CHANGES.md) を参照してください。

---

## English

Modern-IOPaint is a self-hosted image inpainting and outpainting application
that updates [Sanster/IOPaint](https://github.com/Sanster/IOPaint) for current
Python, PyTorch, and Diffusers environments. Its browser UI runs object removal,
masked generation/editing, and illustration repair locally.

### Highlights

- Inpainting and instruction editing with `qwen-image` and `qwen-image-edit`,
  using Nunchaku 1.2.1 4-bit quantization (int4/fp4 selected for compatible
  hardware) and optional 4/8-step Lightning adapters.
- Optional `flux.1-fill-dev` backend. It is governed by the FLUX.1-dev
  Non-Commercial License and requires an in-app license notice and acceptance
  before first use.
- Photo erase models plus `anime-lama` and a dedicated Illustration category.
- Local SD/SDXL `.ckpt` and `.safetensors` loading, with same-name JSON sidecars
  for `prediction_type` and `category`.
- Existing IOPaint model switching, crop/extend masks, helper plugins, and file
  management.

Example local-checkpoint sidecar:

```json
{
  "prediction_type": "v_prediction",
  "category": "inpaint-illustration"
}
```

Place this beside `example.safetensors` as `example.json`.

### Recommended environment

- Windows x64 or Linux x86-64
- An NVIDIA GPU and NVIDIA driver supporting CUDA 12.8
- At least 12 GB VRAM recommended (large images or models may require more)
- At least 32 GB system RAM
- At least 45 GB free disk space
- Python 3.10-3.13 (Python 3.12 is the recommended distribution/verification environment)

The verified dependency tuple is torch `~=2.11.0` + cu128, torchvision
`~=0.26.0`, Diffusers `0.36.0`, Transformers `4.57.6`, and Nunchaku `1.2.1`.

### Install with pip

Run these commands in a fresh virtual environment. `setup-gpu` checks the GPU,
driver, and Python version, then configures CUDA torch/torchvision and the
matching Nunchaku wheel. Nunchaku is deliberately installed with `--no-deps`,
preventing its dependency resolver from replacing CUDA torch with a CPU PyPI
build.

```console
pip install modern-iopaint
modern-iopaint setup-gpu
modern-iopaint start --model qwen-image
```

The default `--device auto` selects `cuda` when available and `cpu` otherwise.
Explicit `--device cpu`, `--device cuda`, and `--device mps` values are still
honored.

To inspect compatibility and planned commands without installing:

```console
modern-iopaint setup-gpu --dry-run
```

### Windows one-click route

Extract the release asset `modern-iopaint-oneclick-<version>.zip` to a writable
folder and double-click `run.bat`. This is an online bootstrap: it fetches the
pinned uv release, uv-managed Python 3.12, the application, and CUDA 12.8
dependencies, then launches `qwen-image` on port 8080. Internet access is
required.

Expect about 4 GB of application/dependency downloads and about 12-40 GB of
model downloads depending on selection. The complete run is written to
`setup.log`. On failure the console remains open so the diagnostic is readable;
running it again repairs or updates the existing environment.

### Models and licenses

| Model/weights | License | Notes |
|---|---|---|
| Qwen Image/Qwen Image Edit, Nunchaku quantizations, Lightning | Apache-2.0 | Used for 4-bit inference; weights come from their upstream publishers. |
| MIGAN | MIT | Erase model; review its upstream license and distribution terms. |
| AnimeLaMa | MIT | Illustration erase model derived from `dreMaz/AnimeMangaInpainting`. |
| FLUX.1-Fill-dev | FLUX.1-dev Non-Commercial License | Optional; non-commercial terms apply and in-app acceptance is required. |
| User-provided local checkpoints | Varies by provider | Whoever provides or uses them is responsible for licenses, rights, and permitted use. |

Neither this repository nor the PyPI wheel bundles model weights.

### Benchmarks

The method and existing reports live in [`benchmarks/`](benchmarks/). Results
depend heavily on hardware, driver, precision, and runtime profile. The formal
release comparison table will be filled after physical-hardware runs.

| GPU | VRAM | Model/configuration | 1024px time | Peak VRAM | Report |
|---|---:|---|---:|---:|---|
| TBD | TBD | Qwen Image int4, 8-step Lightning | TBD | TBD | [`benchmarks/README.md`](benchmarks/README.md) |
| TBD | TBD | FLUX Fill int4 | TBD | TBD | [`benchmarks/README.md`](benchmarks/README.md) |

### Development

Prepare the backend and frontend separately. The frontend must be built before
building a wheel. Hatch directly maps `web_app/dist` into
`modern_iopaint/web_app` inside the wheel, so no manual copy step is needed.

```console
pip install -e .
cd web_app
npm ci
npm run build
cd ..
modern-iopaint start --model lama
```

In a development tree, if package assets are absent, the server detects
`web_app/dist`, logs a notice, and serves it. Use `hatch build` for distribution
artifacts. Before release, populate Nunchaku wheel hashes with
`python scripts/update_wheel_hashes.py` and pin uv with
`python scripts/update_oneclick_pins.py --version <tag>`.

### Credits

This project is based on [IOPaint](https://github.com/Sanster/IOPaint) by
Sanster and the IOPaint contributors. Code is provided under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and
[UPSTREAM_CHANGES.md](UPSTREAM_CHANGES.md) for upstream attribution, the base
commit, and modification history.
