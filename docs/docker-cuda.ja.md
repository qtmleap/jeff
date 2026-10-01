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

## 2. モデルを指定して起動

```sh
git clone --branch feat/cuda-docker https://github.com/qtmleap/jeff.git
cd jeff
docker compose pull jeff
docker compose up -d --no-build --pull never
docker compose logs -f --tail 100 jeff
```

標準モデルは `mstrasser/Jeff-Qwen3.5-2B`。別モデルを指定する場合は同じディレクトリの `.env` に記載します。
モデルは通常のQwenではなく、Jeff用に学習されたチェックポイントを指定してください。

```dotenv
JEFF_MODEL_REPO=mstrasser/Jeff-Qwen3.5-2B
JEFF_MODEL_REVISION=main
```

`JEFF_MODEL_REVISION` は省略可能で、初期値は `main` です。毎回同じモデル版を使いたい場合は
`2b1055eddeb00788f22c0b6156b8d6fa6fc0eecd` のような、そのモデルのコミットSHAを指定してください。
モデル名だけ変更したときに別モデルのSHAを引き継がないよう注意してください。

起動時にHugging Faceから全checkpoint（readout・decision_config・tokenizerを含む）を自動取得し、
UID 10001が書き込める `models` ボリュームに保存します。取得後のsnapshotパスを
`JEFF_CHECKPOINT`へ渡してJeffを起動するため、ホストのモデルパス指定や手動ダウンロードは不要です。
取得失敗や必須ファイル不足なら起動に失敗し、未完成のモデルを使いません。

保存済みblobは再利用し、中断されたダウンロードはHugging Faceの仕組みで再開します。
モデル・revisionごとにsnapshotを分離するため、切り替えても重みを混ぜません。
`main`では起動ごとに更新を確認します。ネットワークなしでキャッシュだけを使用したい場合は
`HF_HUB_OFFLINE: "1"` をサービスのenvironmentへ追加してください。未取得モデルはofflineでは起動できません。
公開モデルを対象とし、保存済み認証トークンは暗黙に使用しません。

初回はモデル約4.2GBの取得が必要です。ログで `Starting Jeff with checkpoint` と起動完了を確認してください。
healthcheckの初期待機期間は30分です。完了を待つ場合は `docker compose up -d --no-build --pull never --wait --wait-timeout 1800`。
回線によってはさらに時間がかかります。healthcheckのunhealthyだけではDockerは再起動しません。

モデル設定を変更した場合は `docker compose up -d --no-build --pull never` を再実行してコンテナを再作成します。

ローカルビルドも可能です。

```sh
export JEFF_IMAGE=jeff-cuda:local
docker compose build jeff
docker compose up -d --no-build --pull never
```

## 3. 接続と検証

既定では `ports` を公開しません。同じComposeネットワークのアプリから **`http://jeff:8765`** へ接続します。
ホストからのlocalhostアクセスはできません。コンテナ内でヘルスと分類を検証できます。

```sh
docker compose exec -T jeff python - <<'PYCODE'
import json, urllib.request
base = "http://127.0.0.1:8765"
print(json.load(urllib.request.urlopen(base + "/health")))
body = {"model": "jeff-latest", "state": "The customer cannot sign in and needs to reset their password.",
        "questions": {"route": {"type": "choice", "criteria": {
            "1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"}}}}
req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=120)))
PYCODE
```

ヘルスがready、選択肢の確率が合計約1、`output_tokens: 0`となることを確認します。
Jeffはチャット生成ではなく分類APIです。初期モデルの主な対象は英語テキストです。

GPUとBF16の確認は次のコマンドです。entrypointを置き換えるのでモデルをダウンロードしません。

```sh
docker compose run --rm --no-deps --entrypoint python jeff -c '
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 unavailable"
x = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(y.device, y.dtype, float(y[0,0]))
'
```

`JEFF_DEVICE=cuda`を明示し、GPUがなければサーバー起動エラーとなります。CPUへはフォールバックしません。
ホストの `nvidia-smi` でサーバープロセスのGPUメモリ使用も確認してください。
初回推論でTriton等のコンパイルが発生する場合があり、コンパイル用キャッシュは別ボリュームに保持します。
VRAM不足の場合は他のGPUプロセスと入力長を確認してください。RTX 5060上の実推論は未検証です。

## 運用

```sh
docker compose stop
docker compose start
docker compose logs -f --tail 100 jeff
docker compose down     # コンテナだけを削除。モデル・コンパイルキャッシュは保持
```

`restart: unless-stopped`を設定しています。ホスト再起動時の復帰にはDockerデーモンの自動起動が必要です。
ログは10MB×3ファイルに制限しています。**`docker compose down -v` は取得済みモデルも削除します。**
モデルとコンパイルキャッシュ以外のルートファイルシステムは読み取り専用です。
既存MacのMLX環境・LaunchAgentは変更しません。

## 検証範囲

起動処理はモデル選択・snapshotパス引き渡し・取得失敗・不完全checkpoint・カスタムコマンドを
ネットワーク境界を置き換えたユニットテストで確認します。このテストはイメージビルド時にも実行します。
実モデルの初回ダウンロードとNVIDIA GPUでのサーバー推論は、上記手順によりUbuntu実機で確認してください。
実行ごとのイメージdigestはGitHub Actionsサマリーを参照してください。

## GHCR公開とビルドキャッシュ

`.github/workflows/publish-image.yaml` は `qtmleap/jeff` の `main` または `feat/cuda-docker` への
ビルド入力変更のpushで実行します。workflow_dispatchも定義していますが、GitHubのUIでの手動実行は
ワークフローがデフォルトブランチに存在することが前提です。featureブランチではpushトリガーを使用します。
mainへのマージはこの設定を追加する作業には含まれません。

利用する標準タグは **`ghcr.io/qtmleap/jeff:latest`** です。
今後の通常ビルドは `latest` と追跡用の `branch-feat-cuda-docker`（mainでは `branch-main`）、
`sha-<完全なコミットSHA>`、上流コード追跡用の `upstream-<git describe>` を公開します。タグは更新可能なので、厳密な固定にはdigestを使います。
`latest`の更新は共有concurrencyグループで直列化します。現在のCUDA対応featureブランチからの更新も対象です。
ワークフローファイルだけの変更では通常ビルドを起動しません。

初回のlatest追加には `.github/workflows/promote-latest.yaml` を使用します。
移行ワークフロー内で固定した確認済みdigestへ `latest` と `upstream-v1.1-1-gf067882` を追加するだけで、再ビルドしません。
この移行ワークフローは自身の変更pushだけで起動し、既存の対象タグが別digestなら上書きせず失敗します。
後日latestが更新された後に古いdigestへ戻す用途には使わないでください。
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

### 上流バージョンの意味

上流の実在リリース [v1.1](https://github.com/firelex/jeff/releases/tag/v1.1) のコミットは
`f0397f3785d93f73a01411d785d2ed026f53181d` です。今回の上流ベースはその1コミット後の
`f06788292874c21a5b5c41549ac220dd9e15da7f`（`git describe`: `v1.1-1-gf067882`）なので、
イメージには **`upstream-v1.1-1-gf067882`** を付け、純粋な `v1.1` タグは付けません。
pyprojectのパッケージ版は `0.2.0` で、上流リリース名やモデルのv1.1とは別のものです。

今後のビルドでは上流mainとの共通祖先を求め、上流タグだけを対象にgit describeして追跡用タグを生成します。
このタグはフォークの上流ベースを表します。Docker構成などフォーク独自の変更で同じ上流タグのdigestが更新される
場合があるため、完全固定が必要な場合はSHAタグに加えてdigestを記録してください。
