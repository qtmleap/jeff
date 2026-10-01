# Ubuntu / RTX 5060 で Jeff を Docker Compose 起動する

対象は **Ubuntu x86_64 + NVIDIA GPU + NVIDIA Container Toolkit + Docker Engine / Compose**。
Mac上のMLX用 `.venv`・モデル・LaunchAgentとは独立した構成です。以下はUbuntu側で実行します。
MacでCUDAイメージを起動・ビルドしないでください。

作成時のJeffコミット: `f06788292874c21a5b5c41549ac220dd9e15da7f`。
Python 3.12、uv 0.12.19、`uv sync --locked --no-default-groups --extra cuda` を使用します。
ロックにはPyTorch 2.14.0、CUDA 13系ライブラリ、Triton、flash-linear-attentionが含まれます。
CUDAのユーザー空間ライブラリはPython依存に含まれるため、ホストへのCUDA Toolkit一式の導入は前提にしません。
ベースイメージはバージョンタグ指定で、digest固定ではありません。

## 1. ホストの前提を確認

```sh
uname -m                         # x86_64
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-ctk --version
docker version
docker compose version
```

ドライバ、Docker、NVIDIA Container Toolkitが未導入なら、Ubuntuの管理者が
[NVIDIA公式導入手順](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
に従って準備してください。DockerへのRuntime設定は同ページの
`sudo nvidia-ctk runtime configure --runtime=docker`、続くDocker再起動が該当します。
既存コンテナへの影響を確認してから実施してください。この構成ファイルはホスト設定を変更しません。

[CUDA互換性表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)では
CUDA 13.xの最小ドライバ系列は **580以上**。ただしこれは最低条件で、Triton/PTX JITや新機能は
より新しいドライバを要求する場合があります。実イメージの `torch.version.cuda` と
[NVIDIAリリースノート](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)も照合してください。
`nvidia-smi`上のCUDA Versionはドライバが対応する上限であり、コンテナ内PyTorchのCUDA版ではありません。

GPUがコンテナに渡ることを先に確認します（小さなUbuntuイメージの取得が発生します）。

```sh
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## 2. ソースとイメージを準備

UbuntuにフォークのCUDA対応ブランチを取得します。既存の `jeff` ディレクトリがある場合は、
別の保存先を指定してください。Macの `.venv`、`.tools`、LaunchAgentはコピー不要です。

```sh
git clone --branch feat/cuda-docker https://github.com/qtmleap/jeff.git
cd jeff
# 想定コミットとローカル変更を確認（既存作業を強制リセットしない）
git rev-parse HEAD
git status --short
# このシェルで後続コマンドも実行する
export JEFF_MODEL_DIR="$HOME/models/Jeff-Qwen3.5-2B"
export JEFF_HTTP_PORT=8766
mkdir -p "$JEFF_MODEL_DIR"
docker compose config --quiet
export JEFF_IMAGE=ghcr.io/qtmleap/jeff:branch-feat-cuda-docker
docker compose pull jeff
```

公開済みイメージを使う場合は上記のpullでビルドを省略できます。運用時はActionsの実行サマリーにある
`ghcr.io/qtmleap/jeff@sha256:...` をJEFF_IMAGEに指定するとdigestで固定できます。
パッケージの公開範囲は組織設定に従います。privateの場合は既存の認可済みレジストリ認証を使用してください。
この手順はPATの新規発行や公開設定変更を要求しません。pull権限がなければ管理者に確認するか、次のローカルビルドを使います。

ローカルでビルドする場合は次に切り替えます。

```sh
export JEFF_IMAGE=jeff-cuda:local
docker compose build jeff
```

初回ビルドはCUDA依存のため数GB以上を取得します。ディスクはモデル約4.2GBに加え、イメージ・
ビルドキャッシュ用に余裕を確保してください。`.dockerignore`は許可したソースだけを送るため、
モデル、仮想環境、Git履歴、通常の秘密情報はbuild contextに入りません。
資格情報をソースコードへ直接書かないでください。

## 3. 固定リビジョンの全モデルを取得

新規または同じモデル専用のディレクトリを使用してください。別モデルの既存ファイルを混ぜないでください。
このダウンロードはGPU不要です。公開モデルなのでトークンは指定しません。

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  --mount "type=bind,src=$JEFF_MODEL_DIR,dst=/download" \
  --entrypoint /app/.venv/bin/hf "$JEFF_IMAGE" \
  download mstrasser/Jeff-Qwen3.5-2B \
  --revision 2b1055eddeb00788f22c0b6156b8d6fa6fc0eecd \
  --local-dir /download

# 全checkpointを取得する。model.safetensorsだけでは起動できない。
for f in model.safetensors readout.safetensors decision_config.json config.json \
         tokenizer.json tokenizer_config.json processor_config.json chat_template.jinja; do
  test -s "$JEFF_MODEL_DIR/$f" || { echo "Missing: $f"; exit 1; }
done
# 公開モデルをコンテナUID 10001から読み取れるようにする
chmod -R a+rX "$JEFF_MODEL_DIR"
printf '%s  %s\n' \
  16a9f5276cb761c7c0a6e4a19de3329a924829e5e381a00337535d45eb70fc5c \
  "$JEFF_MODEL_DIR/model.safetensors" | sha256sum -c -
```

モデルリビジョンは `2b1055eddeb00788f22c0b6156b8d6fa6fc0eecd`。
Composeはモデルを読み取り専用でマウントし、存在しないパスの自動作成を拒否します。

## 4. CUDA・BF16を実機で検証

```sh
docker compose run --rm --no-deps --entrypoint /app/.venv/bin/python jeff -c '
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable; CPU fallback is not allowed"
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Compiled arch list:", torch.cuda.get_arch_list())
assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 unavailable"
x = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(y.device, y.dtype, float(y[0,0]))
'
```

JeffのCUDA実装はBF16を使用します。GPU認識だけでなくBF16演算が通ることが必要です。
`JEFF_DEVICE=cuda`を明示しており、GPUがなければJeffは起動エラーとなります。

## 5. 起動・推論確認

```sh
docker compose up -d --no-build --pull never --wait --wait-timeout 300
docker compose ps
docker compose logs --tail 100 jeff
curl --fail-with-body http://127.0.0.1:8766/health
curl --fail-with-body http://127.0.0.1:8766/v1/systemone \
  -H 'Content-Type: application/json' -d '{
  "model": "jeff-latest",
  "state": "The customer cannot sign in and needs to reset their password.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"}
    }
  }
}'
```

`/health`は `status: ready`、`model: jeff-qwen3.5-2b` を確認します。
推論は3選択肢の確率（合計約1）、`choice`、`confidence`、`output_tokens: 0` を返すことを確認します。
この例では通常 `choice: "3"` が期待されますが、出力値の完全一致は要求しません。
チャット生成APIではなく分類APIです。公式モデルの主な対象は英語テキストです。

別ターミナルで `watch -n 1 nvidia-smi` を実行し、起動後のPythonプロセス・GPUメモリ使用と
リクエスト中のGPU使用率を確認します。短い推論はサンプリングで見逃すことがあるため数回送ってください。
`/health`単独では使用GPUを証明しません。上記CUDA/BF16検証、強制CUDA設定、実推論を合わせて確認します。

初回推論ではTriton等のコンパイルに時間がかかる場合があります。UID 10001が書き込める
`compile-cache`ボリュームにHF・Triton・Torch・CUDAキャッシュを保存します。
モデルはローカルですが、kernels等が初回に追加コードを取得する場合があるため、完全オフライン動作は未保証です。
VRAM不足の場合は他のGPUプロセスを確認し、短い入力・単一リクエストで再検証してください。
RTX 5060実機のVRAM余裕、対応カーネル、速度は現時点では未検証です。

## 運用

```sh
docker compose stop                  # 停止。自動再起動を抑制
docker compose start                 # 再開
docker compose restart jeff          # 再起動
docker compose logs -f --tail 100 jeff
docker compose down                  # コンテナ削除。モデルとキャッシュは保持
```

`restart: unless-stopped`によりプロセス終了やDocker再起動後に復帰します（手動停止を除く）。
ホスト起動時の復帰にはDockerデーモン自体の自動起動が必要です。
healthcheckがunhealthyになっただけではDockerは再起動しません。ログを確認してください。
ログは10MB×3ファイルに制限しています。キャッシュを消したい場合だけ `docker compose down -v` を使います。

公開先の初期値は **127.0.0.1:8766** です。コンテナ内は0.0.0.0:8765ですが、LANへは公開しません。
MacのMLXサービスは127.0.0.1:8765のままで、変更しません。
環境変数はシェルを閉じると消えるため、次回もJEFF_MODEL_DIRとJEFF_IMAGEを設定してください。
認証・外部公開はこの構成に含めていません。

## 検証範囲

2026-10-01、Mac上で `docker compose config --quiet` と解決済みJSONの検証を実施済みです。
GPU予約、ループバック公開、モデル読み取り専用、キャッシュ、再起動設定、
モデルパス未指定時の拒否、全シェル例の `sh -n` とBF16検証コードのPython構文を確認しました。
Dockerfileは起動・依存・非root・healthcheck設定を静的確認しました。Hadolintは未導入のため未実行です。
ビルドチェックによるイメージ検証も未実施です。既存のMLX環境には変更を加えていません。
Ubuntu用イメージのビルド、NVIDIA Container Toolkit連携、CUDA/BF16実演算、GPU推論、再起動復帰は
Ubuntu実機での上記検証が必要です。MacのMLX検証結果をCUDA検証結果として扱わないでください。

参考: [Docker Compose GPU予約](https://docs.docker.com/compose/how-tos/gpu-support/)、
[Jeff](https://github.com/firelex/jeff)、
[モデルカード](https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B)。

## GHCR公開とビルドキャッシュ

`.github/workflows/publish-image.yaml` は `qtmleap/jeff` の `main` または `feat/cuda-docker` への
ビルド入力変更のpushで実行します。workflow_dispatchも定義していますが、GitHubのUIでの手動実行は
ワークフローがデフォルトブランチに存在することが前提です。featureブランチではpushトリガーを使用します。
mainへのマージはこの設定を追加する作業には含まれません。

GHCRタグは `branch-feat-cuda-docker`（mainでは `branch-main`）と `sha-<完全なコミットSHA>`。
`latest`は作りません。SHAタグはコミット識別用ですが、レジストリ側の不変性保証はないため、厳密な固定にはdigestを使います。
権限は `contents: read` と公開ジョブの `packages: write`、認証は標準 `GITHUB_TOKEN` のみです。
パッケージ公開範囲・組織ポリシーは変更しません。認証情報をbuild argやイメージへ渡しません。

Dockerfileは依存定義を先にコピーし、`--no-install-project` でCUDA依存レイヤーを作ります。
ソースをコピーした後にプロジェクトをインストールするため、ソースだけの変更では依存レイヤーを再利用します。
どちらのsyncも `--locked` を維持します。uvは `--mount=type=cache` を使い、同じBuildKit上でダウンロードを再利用します。
Actions間では `cache-from/cache-to: type=gha,mode=max` によりレイヤーを再利用します。
uvのcache mountの中身自体はGHAへexportされないため、依存レイヤーが無効になると新規runnerでは再ダウンロードします。
大きなCUDA wheelキャッシュの二重保存を避けるため、cache mountの別途アップロードは行いません。

標準の使い捨てUbuntu runnerで不要なプリインストールSDKを削除して空き容量を確保します。
モデル重みやローカル環境はbuild contextに含まず、公開イメージ・Actionsキャッシュにも含めません。
runnerではCUDA版PyTorchとJeffのimportをビルド時に確認しますが、GPU推論は実機で別途検証してください。
