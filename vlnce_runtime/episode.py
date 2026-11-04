"""Shared R2R episode selection, public observations and private evidence."""

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
MAX_STEPS = 500
SUCCESS_DISTANCE = 3.0


def flag(value, name):
    if isinstance(value, (bool, Real)) and value in (0, 1):
        return bool(value)
    raise ValueError("missing or invalid simulator " + name)


def dataset_paths(data_root, split):
    """Accept the official data/datasets layout or an explicit flat dataset root."""
    root = Path(data_root)
    candidates = [base / "R2R_VLNCE_v1-3" / split / (split + ".json.gz") for base in (root, root / "datasets")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("require exactly one R2R dataset: " + ", ".join(str(path) for path in candidates))
    dataset = found[0]
    gt = dataset.parents[2] / "R2R_VLNCE_v1-3_preprocessed" / split / (split + "_gt.json.gz")
    if not gt.is_file():
        raise FileNotFoundError("missing R2R reference trajectories: " + str(gt))
    return dataset, gt


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
        raise ValueError("R2R episode does not contain a navigation instruction")
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

    def reset(self, params):
        self.close()
        if set(params) != {"vlnce_path", "eval_set", "task_index", "resolution"}:
            raise ValueError("reset requires vlnce_path, eval_set, task_index and resolution")
        if Path(params["vlnce_path"]).resolve() != self.root:
            raise ValueError("reset vlnce_path differs from the launched checkout")
        if params["eval_set"] != self.profile.get("eval_set", "val_unseen"):
            raise ValueError("split differs from the explicitly launched runtime profile")
        index, resolution = params["task_index"], params["resolution"]
        if type(resolution) is not int or not 32 <= resolution <= 2048:
            raise ValueError("resolution must be an integer in [32, 2048]")
        dataset, gt = dataset_paths(self.profile["data_root"], params["eval_set"])
        raw = dataset.read_bytes()
        payload = json.loads(gzip.decompress(raw).decode("utf-8"))
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if not isinstance(episodes, list) or type(index) is not int or not 0 <= index < len(episodes):
            raise ValueError("task_index is outside the selected dataset")
        episode = episodes[index]
        scene = scene_path(episode["scene_id"], self.profile["data_root"])
        self.text = instruction(episode)
        self.native = self.create_environment(episode, payload, dataset, gt, scene, resolution)
        self.state = {
            "episode_id": Path.cwd().name,
            "vlnce_episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "eval_set": params["eval_set"],
            "task_index": index,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_path_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(),
            "runtime_config_sha256": self.configuration_sha256,
            "simulator_backend": self.profile["backend"],
            "simulator_versions": self.versions,
            "env_step": 0,
            "task_success": False,
            "terminated": False,
            "truncated": False,
            "max_episode_steps": MAX_STEPS,
            "action_count": len(ACTIONS),
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
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError("action must be an integer in [0, 3]")
        if self.state["terminated"] or self.state["truncated"]:
            raise RuntimeError("episode has ended")
        before = list(self.agent_state().position)
        observations, done, info = self.advance(HABITAT_ACTIONS[action])
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
        feedback = (
            "Moved forward 0.25 m." if action_success else "Blocked by an obstacle; did not move.",
            "Turned left 15 degrees.",
            "Turned right 15 degrees.",
            "Stop issued; the episode has ended.",
        )[action]
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
            "available_actions": dict(enumerate(ACTIONS)),
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
