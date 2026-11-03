# Standalone R2R runtime

This directory adapts the upstream environment, not its trainers. `launch-worker.sh`
selects an explicitly configured Python; `worker.py` serves JSONL; `episode.py`
owns public action/reward/observation semantics; the two backend modules isolate
Habitat version-specific APIs. No module imports BetaEngine.

Copy `environments/runtime.example.json` to `.vlnce/runtime.json` and configure:

| Field | Meaning |
|---|---|
| `version` | Configuration schema version, currently `1` |
| `environments.r2r_unseen.backend` | Explicit `habitat017` or `habitat033`; no fallback |
| `prefix` | Existing interpreter environment; executable is `<prefix>/bin/python` |
| `data_root` | Contains `R2R_VLNCE_v1-3/` and `R2R_VLNCE_v1-3_preprocessed/`, directly or under `datasets/`, plus `scene_datasets/` or `mp3d_habitat/` |
| `gpu_device_id` | Simulator GPU index, default `0` |
| `env` | Child-only environment variables, including any required rendering/library settings |

Paths are relative to the runtime configuration file. The configuration is ignored
by Git. Do not copy credentials into examples or publish local runtime files.
The launcher uses `python3` for its standard-library bootstrap, then execs the
configured interpreter with its `bin` directory on PATH. It does not activate or
install Conda environments: supply required native-library settings explicitly.

For `habitat017`, prepare a **separate** Python 3.6 environment using the upstream
[installation instructions](../README.md#setup): Habitat-Sim and Habitat-Lab
must both be version 0.1.7. Dataset/extension dependencies must be available there.
The adapter calls `habitat.Env`, not a trainer. It disables oracle policy sensors
and publishes RGB only, even though native RGBD sensors may be configured.

`habitat033` retains the explicit `NavRLEnv` port interface and requires
`vlnce_baselines/common/nav_rl_env_habitat033.py`. **That compatibility implementation
is not included in the upstream revision used here.** Supply a reviewed, licensed,
revision-bound implementation before enabling this profile. The adapter reports
the missing file rather than substituting the 0.1.7 backend. It is not a claim
that a complete 0.3.x deployment has been validated.

Create an episode directory, then launch:

```bash
mkdir -p /tmp/vlnce-episode
./launch-worker.sh --env r2r_unseen --work-dir /tmp/vlnce-episode \
  --runtime-config /absolute/path/runtime.json
```

Protocol:

```json
{"id":1,"method":"reset","params":{"vlnce_path":"/absolute/path/VLN-CE","eval_set":"val_unseen","task_index":0,"resolution":128}}
{"id":2,"method":"step","params":{"action":3}}
{"id":3,"method":"close","params":{}}
```

Responses have `id`, `ok` and either `result` or `error/message`. `result` contains
`observation/reward/state`. Stdout is protocol-only; diagnostics use stderr.
Public actions are forward 0.25 m, left 15 degrees, right 15 degrees, STOP.
Reward is native successful STOP = 1, otherwise 0. Failed STOP terminates;
500 non-STOP steps truncate. Private evidence binds the actual checkout,
adapter bytes, runtime configuration, dataset and reference trajectories.
Different backend versions are separate experimental protocols, not equivalent scores.

Offline tests (in an environment with pytest, NumPy and Pillow):

```bash
python -m pytest -q vlnce_runtime/tests
```

These are synthetic contract tests, not simulator acceptance. Also execute the
launcher/close protocol under the configured minimum Python version and perform
real reset, image hash, action, STOP and cleanup checks before publishing results.

## Installation and native demonstrations

With a modern bootstrap Python, `python vlnce_runtime/install.py --conda /path/to/conda`
creates an isolated official 0.1.7 environment, downloads public R2R assets and writes
`.vlnce/runtime.json`. Use `--dry-run` to inspect commands, `check` to inspect an existing
runtime, and `--prefix`/`--data-root` to reuse local resources. Existing external prefixes
are not upgraded. MP3D scene access follows the official dataset authorization process.

`demo.py` runs under the configured simulator Python and collects a successful native
ShortestPathFollower episode in a new output directory. It retains the original transitions
and publishes `demo.json` only after confirmed success ending in STOP. This privileged
oracle is a separate development command, never a worker tool or policy observation.
BetaEngine can replay its actions in a fresh episode with identity checks and live images.
