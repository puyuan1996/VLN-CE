# Standalone R2R / RxR runtime

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

## Training split

`--env r2r_train` selects `environments.r2r_train` and accepts only `eval_set=train`.
The existing `r2r_unseen` profile remains restricted to `val_unseen`. Both profiles
use the same public actions and native STOP success rule. Configure them explicitly;
no profile fallback is performed. `install.py` and `demo.py` accept `--eval-set train`.
The example runtime JSON contains both profiles; they can share an interpreter and data root.
Dataset and NDTW paths follow the selected split. Oracle sensors remain disabled for
policy execution, and the separate demo command remains privileged.

## RxR and scene splits

Profiles are `{r2r,rxr}_{train,seen,unseen}`, fixing `train`, `val_seen`, or `val_unseen`. `install.py --dataset rxr` downloads the official `RxR_VLNCE_v0` archive and fills missing profiles, preserving other existing profiles. Both datasets may share an interpreter and scene directory.

RxR reset additionally accepts `role` (`guide` or `follower`) and `languages` (`["*"]` or explicit en-US/en-IN/hi-IN/te-IN codes). The task index follows source order after language filtering. Native RxR dataset parsing receives its split config without eagerly loading every episode. BERT features and oracle sensors are not policy dependencies.

RxR uses forward 0.25 m, left/right 30°, STOP, look up/down 30° at indices 0..5. Its native RGB camera uses 640×480, HFOV 79°; reset resolution sets width and preserves 4:3. Only RGB is published. Dataset paths are `RxR_VLNCE_v0/{split}/{split}_{role}.json.gz` with sibling `{split}_{role}_gt.json.gz`. `habitat033` has no verified RxR bridge and is rejected.

Optional `expected_dataset_sha256` and `expected_reference_path_sha256` reset fields reject changed frozen inputs before creating a simulator. Selected episodes must have a native reference trajectory. Both datasets retain STOP success reward 0/1 and the 500-step limit. Test/challenge splits without local goals are deliberately excluded from scored training.
