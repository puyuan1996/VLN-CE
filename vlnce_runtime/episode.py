"""R2R/RxR episode selection, public observations and private native evidence."""

import base64
import gzip
import hashlib
import importlib
import json
import math
import subprocess
from numbers import Real
from pathlib import Path

ACTIONS = ("move forward 0.25m", "turn left 15 degrees", "turn right 15 degrees", "stop")
HABITAT_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
RXR_ACTIONS = ("move forward 0.25m", "turn left 30 degrees", "turn right 30 degrees", "stop",
               "look up 30 degrees", "look down 30 degrees")
MAX_STEPS = 500
SUCCESS_DISTANCE = 3.0


def flag(value, name):
    if isinstance(value, (bool, Real)) and value in (0, 1):
        return bool(value)
    raise ValueError("missing or invalid simulator " + name)


def dataset_paths(data_root, split, dataset="r2r", role="guide"):
    """Accept the official data/datasets layout or an explicit flat dataset root."""
    if dataset not in ("r2r", "rxr") or split not in ("train", "val_seen", "val_unseen"):
        raise ValueError("select r2r/rxr and a scored train/val_seen/val_unseen split")
    if role not in ("guide", "follower"):
        raise ValueError("RxR role must be guide or follower")
    root = Path(data_root)
    directory = "R2R_VLNCE_v1-3" if dataset == "r2r" else "RxR_VLNCE_v0"
    filename = split if dataset == "r2r" else split + "_" + role
    candidates = [base / directory / split / (filename + ".json.gz") for base in (root, root / "datasets")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("require exactly one " + dataset + " dataset: " + ", ".join(str(path) for path in candidates))
    path = found[0]
    gt = (path.parents[2] / "R2R_VLNCE_v1-3_preprocessed" / split / (split + "_gt.json.gz")
          if dataset == "r2r" else path.with_name(filename + "_gt.json.gz"))
    if not gt.is_file():
        raise FileNotFoundError("missing reference trajectories: " + str(gt))
    return path, gt


def select_episodes(payload, dataset="r2r", languages=None):
    """Filter RxR in source order; task_index is relative to this explicit selection."""
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset requires nonempty episodes")
    if dataset == "rxr":
        languages = ["*"] if languages is None else languages
        if (not isinstance(languages, list) or not languages
                or any(language not in ("*", "en-US", "en-IN", "hi-IN", "te-IN") for language in languages)
                or len(set(languages)) != len(languages) or ("*" in languages and len(languages) != 1)):
            raise ValueError("languages must select '*' or unique RxR language codes")
        if "*" not in languages:
            episodes = [row for row in episodes if row["instruction"].get("language") in languages]
        if not episodes:
            raise ValueError("no RxR episodes match the selected languages")
    identifiers = [str(row["episode_id"]) for row in episodes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate native episode IDs within one dataset role/split")
    return episodes


def scene_path(scene_id, data_root):
    candidate = Path(scene_id)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError("missing absolute scene path: " + str(candidate))
    relative = str(scene_id)
    prefix = "data/scene_datasets/"
    if relative.startswith(prefix):
        relative = relative[len(prefix) :]
    roots = [Path(data_root) / name for name in ("scene_datasets", "mp3d_habitat")]
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("scene not found under configured data_root: " + str(scene_id))


def instruction(episode):
    value = episode.get("instruction")
    if isinstance(value, dict):
        value = value.get("instruction_text") or value.get("text")
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    if not isinstance(value, str) or not value.strip():
        value = episode.get("instruction_text") or episode.get("instructions")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("VLN-CE episode does not contain a navigation instruction")
    return value


def source_identity(root):
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=5,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=5,
    )
    digest = hashlib.sha256()
    paths = (
        list((root / "vlnce_runtime").glob("*.py"))
        + list((root / "habitat_extensions").rglob("*.py"))
        + list((root / "habitat_extensions/config").glob("*.yaml"))
        + [root / "launch-worker.sh"]
    )
    bridge = root / "vlnce_baselines/common/nav_rl_env_habitat033.py"
    if bridge.is_file():
        paths.append(bridge)
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0" + path.read_bytes())
    return {
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "adapter_sha256": digest.hexdigest(),
    }


class NavigationBackend:
    """Keep task semantics and observation delivery identical across version adapters."""

    def __init__(self, root, profile, configuration_sha256):
        self.root = root
        self.profile = profile
        self.configuration_sha256 = configuration_sha256
        self.native = None
        self.state = {}
        self.dataset = profile.get("dataset", "r2r")
        self.actions = RXR_ACTIONS if self.dataset == "rxr" else ACTIONS
        self.native_actions = HABITAT_ACTIONS + (("LOOK_UP", "LOOK_DOWN") if self.dataset == "rxr" else ())

    def reset(self, params):
        required = {"vlnce_path", "eval_set", "task_index", "resolution"}
        optional = {"expected_dataset_sha256", "expected_reference_path_sha256"}
        if self.dataset == "rxr":
            optional |= {"role", "languages"}
        if not required <= set(params) or set(params) - required - optional:
            raise ValueError("reset requires vlnce_path, eval_set, task_index and resolution; RxR also accepts role/languages")
        if Path(params["vlnce_path"]).resolve() != self.root:
            raise ValueError("reset vlnce_path differs from the launched checkout")
        if params["eval_set"] != self.profile.get("eval_set", "val_unseen"):
            raise ValueError("split differs from the explicitly launched runtime profile")
        index, resolution = params["task_index"], params["resolution"]
        if type(resolution) is not int or not 32 <= resolution <= 2048:
            raise ValueError("resolution must be an integer in [32, 2048]")
        role, languages = params.get("role", "guide"), params.get("languages", ["*"])
        dataset, gt = dataset_paths(self.profile["data_root"], params["eval_set"], self.dataset, role)
        raw = dataset.read_bytes()
        reference_raw = gt.read_bytes()
        checksums = {"dataset_sha256": hashlib.sha256(raw).hexdigest(),
                     "reference_path_sha256": hashlib.sha256(reference_raw).hexdigest()}
        for key, checksum in checksums.items():
            if params.get("expected_" + key, checksum) != checksum:
                raise ValueError("frozen " + key + " differs from the current dataset")
        payload = json.loads(gzip.decompress(raw).decode("utf-8"))
        episodes = select_episodes(payload, self.dataset, languages)
        if type(index) is not int or not 0 <= index < len(episodes):
            raise ValueError("task_index is outside the selected dataset")
        episode = episodes[index]
        references = json.loads(gzip.decompress(reference_raw).decode("utf-8"))
        reference = references.get(str(episode["episode_id"]), {}).get("locations")
        if not reference:
            raise ValueError("selected episode has no native reference trajectory")
        scene = scene_path(episode["scene_id"], self.profile["data_root"])
        text = instruction(episode)
        self.close()
        self.text = text
        self.selection = {"role": role, "languages": languages} if self.dataset == "rxr" else {}
        self.native = self.create_environment(episode, payload, dataset, gt, scene, resolution)
        self.state = {
            "episode_id": Path.cwd().name,
            "vlnce_episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "eval_set": params["eval_set"],
            "task_index": index,
            "dataset": self.dataset,
            **self.selection,
            **({"instruction_id": episode["instruction"].get("instruction_id"),
                "language": episode["instruction"].get("language"),
                "trajectory_id": episode.get("trajectory_id")} if self.dataset == "rxr" else {}),
            **checksums,
            "runtime_config_sha256": self.configuration_sha256,
            "simulator_backend": self.profile["backend"],
            "simulator_versions": self.versions,
            "env_step": 0,
            "task_success": False,
            "terminated": False,
            "truncated": False,
            "max_episode_steps": MAX_STEPS,
            "action_count": len(self.actions),
            "resolution": resolution,
            "seed_applied": False,
            "state_known": True,
            **source_identity(self.root),
        }
        obs = self.native.reset()
        self._private_state(self.metrics())
        return self._result(obs, 0.0, "Initial camera observation.")

    def step(self, action):
        if self.native is None:
            raise RuntimeError("reset before step")
        if type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError("action must be an integer in [0, %d]" % (len(self.actions) - 1))
        if self.state["terminated"] or self.state["truncated"]:
            raise RuntimeError("episode has ended")
        before = list(self.agent_state().position)
        observations, done, info = self.advance(self.native_actions[action])
        position = self.agent_state().position
        moved = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(position, before)))
        self.state["env_step"] += 1
        success = flag(info.get("success"), "success")
        if success and action != 3:
            raise ValueError("native success must require an explicit STOP")
        budget = self.state["env_step"] >= MAX_STEPS and action != 3
        terminated = action == 3
        if flag(done, "done") and not (terminated or budget):
            raise ValueError("simulator ended without a known termination or truncation reason")
        action_success = moved > 1e-6 if action == 0 else True
        self.state.update(
            task_success=success,
            action_success=action_success,
            native_done=bool(done),
            terminated=terminated,
            truncated=budget,
            termination_reason=("task_success" if success else "task_failure") if terminated else None,
            truncation_reason="time_limit" if budget else None,
        )
        self._private_state(info)
        feedback = self.actions[action].capitalize() + "."
        if action == 0:
            feedback = "Moved forward %.3f m." % moved if action_success else "Blocked by an obstacle; did not move."
        elif action == 3:
            feedback = "Stop issued; the episode has ended."
        return self._result(observations, 1.0 if success else 0.0, feedback)

    def _private_state(self, info):
        for key in ("distance_to_goal", "path_length"):
            if key not in info or not isinstance(info[key], Real) or not math.isfinite(info[key]):
                raise ValueError("missing or nonfinite native metric: " + key)
        state = self.agent_state()
        rotation = getattr(state.rotation, "components", state.rotation)
        self.state.update(
            position=[float(value) for value in state.position],
            rotation=[float(value) for value in rotation],
            **{
                key: float(info[key])
                for key in ("distance_to_goal", "path_length", "oracle_success", "spl", "ndtw")
                if key in info
            },
        )

    def _result(self, observations, reward, feedback):
        image_module = importlib.import_module("PIL.Image")
        frame = observations["rgb"]
        if len(frame.shape) != 3 or frame.shape[2] != 3 or str(frame.dtype) != "uint8":
            raise ValueError("rgb must be a uint8 RGB array")
        path = Path.cwd() / ("frame-%03d.png" % self.state["env_step"])
        image_module.fromarray(frame).save(str(path))
        raw = path.read_bytes()
        image = {
            "path": str(path),
            "mime_type": "image/png",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "step": self.state["env_step"],
            "shape": list(frame.shape),
            "dtype": "uint8",
        }
        visible = {
            "instruction": self.text,
            "feedback": feedback,
            "env_step": self.state["env_step"],
            "episode_ended": self.state["terminated"] or self.state["truncated"],
            "available_actions": dict(enumerate(self.actions)),
        }
        marker = "[IMAGE:data:image/png;base64," + base64.b64encode(raw).decode("ascii") + "]"
        return {
            "observation": {"content": json.dumps(visible, ensure_ascii=False) + "\n" + marker, "images": [image]},
            "reward": reward,
            "state": dict(self.state),
        }

    def close(self):
        if self.native is not None:
            native, self.native = self.native, None
            native.close()
