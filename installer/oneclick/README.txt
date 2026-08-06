Modern-IOPaint Windows ワンクリック・オンラインブートストラップ
================================================================

これはオフラインインストーラーではありません。run.bat をダブルクリックすると、
インターネットから uv、Python 3.12、Modern-IOPaint、CUDA 12.8 用 PyTorch、
Nunchaku、および選択したモデルをダウンロードします。

Modern-IOPaint 本体の wheel は daraskme/Modern-IOpaint の最新 GitHub Release
から取得し、依存関係は PyPI から取得します。Modern-IOPaint 本体の PyPI からの
インストールは、公開準備が完了した後に提供予定です。

必要条件:
- Windows x64 と NVIDIA GPU
- CUDA 12.8 をサポートする NVIDIA ドライバー
- 45 GB 以上の空き容量
- インターネット接続

ダウンロード量の目安は、アプリと依存関係が約 4 GB、モデルが選択内容により
約 12～40 GB です。初回起動には時間がかかります。

使い方:
1. この ZIP を書き込み可能なフォルダーへ展開します。
2. run.bat をダブルクリックします。
3. セットアップ完了後、ブラウザーで http://localhost:8080 が開きます。

再実行しても既存の環境を利用し、壊れた/古いパッケージを修復・更新します。
失敗時は画面のメッセージと setup.log を確認してください。

開発者向けローカル wheel テスト:
  run.bat -LocalWheel "C:\path\to\modern_iopaint-<version>-py3-none-any.whl"


Modern-IOPaint Windows one-click online bootstrap
==================================================

This is not an offline installer. Double-clicking run.bat downloads uv,
Python 3.12, Modern-IOPaint, CUDA 12.8 PyTorch, Nunchaku, and the selected
models from the internet.

The Modern-IOPaint wheel is installed from the latest GitHub Release for
daraskme/Modern-IOpaint, while dependencies still come from PyPI. Installing
Modern-IOPaint itself from PyPI will be available later, after publishing is ready.

Requirements:
- 64-bit Windows and an NVIDIA GPU
- An NVIDIA driver that supports CUDA 12.8
- At least 45 GB free disk space
- An internet connection

Expected downloads are about 4 GB for the application and dependencies, plus
about 12-40 GB for models depending on your selection. The first launch can
take a while.

Usage:
1. Extract this ZIP to a writable folder.
2. Double-click run.bat.
3. After setup, http://localhost:8080 opens in your browser.

It is safe to run the bootstrap again: it reuses the environment and repairs
or updates outdated packages. If setup fails, read the console message and
setup.log.

Local wheel test for maintainers:
  run.bat -LocalWheel "C:\path\to\modern_iopaint-<version>-py3-none-any.whl"
