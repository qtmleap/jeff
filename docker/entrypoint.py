"""Download the selected Jeff snapshot, then replace this process with the server."""
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


def main(argv: list[str]) -> None:
    command = argv or ['/app/.venv/bin/jeff-serve']
    environment = os.environ.copy()
    if Path(command[0]).name == 'jeff-serve':
        repo = environment.get('JEFF_MODEL_REPO', 'mstrasser/Jeff-Qwen3.5-2B')
        revision = environment.get('JEFF_MODEL_REVISION', 'main')
        cache = environment.get('JEFF_MODEL_CACHE', '/models')
        print(f'Preparing Jeff model {repo} at revision {revision}', flush=True)
        # Full snapshots include the trained readout and decision config, not only backbone weights.
        # Hub caching isolates repositories/revisions, resumes partial downloads and reuses existing blobs.
        checkpoint = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, token=False)
        for name in ('config.json', 'decision_config.json', 'readout.safetensors'):
            path = Path(checkpoint) / name
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f'Incomplete Jeff checkpoint: missing or empty {name}')
        environment['JEFF_CHECKPOINT'] = checkpoint
        print(f'Starting Jeff with checkpoint {checkpoint}', flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as error:
        print(f'Jeff startup failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
