"""Offline protocol and native adapter checks; never impersonate a real simulator smoke."""

import ast
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vlnce_runtime.configuration import load_profile
from vlnce_runtime.episode import NavigationBackend, dataset_paths
from vlnce_runtime.habitat017 import Habitat017Backend
from vlnce_runtime.habitat033 import Habitat033Backend

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def configured(tmp_path):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to(sys.executable)
    data = tmp_path / "data"
    data.mkdir()
    profile = {"backend": "habitat017", "prefix": str(prefix), "data_root": str(data)}
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"version": 1, "environments": {"r2r_unseen": profile}}))
    return config, profile


def test_launcher_uses_explicit_interpreter_and_relative_config(configured, tmp_path):
    config, _ = configured
    work = tmp_path / "episode"
    work.mkdir()
    result = subprocess.run(
        [
            str(ROOT / "launch-worker.sh"),
            "--env",
            "r2r_unseen",
            "--work-dir",
            str(work),
            "--runtime-config",
            config.name,
        ],
        cwd=str(tmp_path),
        input='{"id":1,"method":"close","params":{}}\n',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"id": 1, "ok": True, "result": {}}


@pytest.mark.parametrize("backend", [None, "auto", "habitat017,habitat033"])
def test_backend_is_explicit_without_fallback(configured, backend):
    config, profile = configured
    profile["backend"] = backend
    config.write_text(json.dumps({"version": 1, "environments": {"r2r_unseen": profile}}))
    with pytest.raises(ValueError, match="explicitly select"):
        load_profile(config)


def test_configuration_is_hash_bound_and_error_response_is_diagnostic(configured, tmp_path):
    config, _ = configured
    profile, checksum = load_profile(config)
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="changed during launch"):
        load_profile(config, expected_sha256=checksum)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "vlnce_runtime/worker.py"),
            "--runtime-config",
            str(config),
            "--runtime-sha256",
            checksum,
        ],
        cwd=str(tmp_path),
        input='{"id":2,"method":"reset","params":{}}\n',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=10,
    )
    response = json.loads(result.stdout)
    assert response["id"] == 2 and response["ok"] is False
    assert "changed during launch" in response["message"]
    assert "Traceback" in result.stderr and "Traceback" not in result.stdout


@pytest.fixture
def episode_files(configured):
    config, profile = configured
    root = Path(profile["data_root"])
    dataset = root / "datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz"
    gt = root / "datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz"
    for path in (dataset, gt):
        path.parent.mkdir(parents=True)
    scene = root / "scene_datasets/mp3d/scene/scene.glb"
    scene.parent.mkdir(parents=True)
    scene.touch()
    episodes = [
        {
            "episode_id": str(index),
            "scene_id": "mp3d/scene/scene.glb",
            "instruction": {"instruction_text": "find room " + str(index)},
            "start_position": [0, 0, 0],
            "start_rotation": [0, 0, 0, 1],
            "goals": [{"position": [1, 0, 0]}],
        }
        for index in (11, 7)
    ]
    payload = {"instruction_vocab": {"word_list": ["find"]}, "episodes": episodes}
    dataset.write_bytes(gzip.compress(json.dumps(payload).encode()))
    gt.write_bytes(gzip.compress(b"{}"))
    profile["gpu_device_id"] = 0
    return profile, dataset, gt, scene, payload


class FixtureBackend(NavigationBackend):
    def create_environment(self, episode, payload, dataset, gt, scene, resolution):
        self.selected = episode
        self.versions = {"fixture": True}
        self.position = [0, 0, 0]
        self.success = False
        self.native = SimpleNamespace(reset=lambda: self.frame(), close=lambda: None)
        return self.native

    def frame(self):
        return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    def metrics(self):
        return {"success": self.success, "distance_to_goal": 1, "path_length": 0}

    def agent_state(self):
        return SimpleNamespace(position=self.position, rotation=[0, 0, 0, 1])

    def advance(self, action):
        self.last_action = action
        if action == "MOVE_FORWARD":
            self.position[0] += 0.25
        return self.frame(), action == "STOP", self.metrics()


def test_episode_selection_actions_private_evidence_and_failed_stop(episode_files, tmp_path, monkeypatch):
    profile, dataset, _, _, _ = episode_files
    monkeypatch.chdir(tmp_path)
    backend = FixtureBackend(ROOT, profile, "configuration")
    result = backend.reset({"vlnce_path": str(ROOT), "eval_set": "val_unseen", "task_index": 1, "resolution": 128})
    assert result["state"]["vlnce_episode_id"] == "7"
    assert result["state"]["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["state"]["source_revision"]
    assert result["state"]["adapter_sha256"]
    visible = result["observation"]["content"]
    assert "find room 7" in visible
    assert "distance_to_goal" not in visible and "position" not in visible
    invalid_state = dict(backend.state)
    with pytest.raises(ValueError, match="integer"):
        backend.step(True)
    assert backend.state == invalid_state
    for index, name in enumerate(("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")):
        result = backend.step(index)
        assert backend.last_action == name and not result["state"]["terminated"]
    result = backend.step(3)
    assert backend.last_action == "STOP" and result["reward"] == 0
    assert result["state"]["termination_reason"] == "task_failure"
    assert result["state"]["terminated"] and not result["state"]["truncated"]
    image = result["observation"]["images"][0]
    assert hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest() == image["sha256"]
    with pytest.raises(RuntimeError, match="ended"):
        backend.step(0)
    backend.close()


def test_success_and_time_limit_remain_distinct(episode_files, tmp_path, monkeypatch):
    profile, _, _, _, _ = episode_files
    monkeypatch.chdir(tmp_path)
    backend = FixtureBackend(ROOT, profile, "configuration")
    params = {"vlnce_path": str(ROOT), "eval_set": "val_unseen", "task_index": 0, "resolution": 128}
    backend.reset(params)
    backend.state["env_step"] = 499
    result = backend.step(1)
    assert result["state"]["truncated"] and not result["state"]["terminated"]
    assert result["state"]["truncation_reason"] == "time_limit"
    backend.reset(params)
    backend.success = True
    result = backend.step(3)
    assert result["reward"] == 1 and result["state"]["termination_reason"] == "task_success"
    assert result["state"]["terminated"] and not result["state"]["truncated"]
    backend.close()


def test_official_adapter_uses_habitat_env_and_selected_episode(episode_files, monkeypatch):
    profile, dataset, gt, scene, payload = episode_files
    config = SimpleNamespace(
        defrost=lambda: None,
        freeze=lambda: None,
        ENVIRONMENT=SimpleNamespace(),
        DATASET=SimpleNamespace(TYPE="VLN-CE-v1"),
        TASK=SimpleNamespace(NDTW=SimpleNamespace()),
        SIMULATOR=SimpleNamespace(
            HABITAT_SIM_V0=SimpleNamespace(), RGB_SENSOR=SimpleNamespace(), DEPTH_SENSOR=SimpleNamespace()
        ),
    )
    captured = {}

    class Dataset:
        def from_json(self, raw):
            captured["payload"] = json.loads(raw)
            self.episodes = [SimpleNamespace(**captured["payload"]["episodes"][0])]

    def create_env(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "habitat",
        SimpleNamespace(
            __version__="0.1.7",
            __file__="/fixture/habitat/__init__.py",
            make_dataset=lambda name: Dataset(),
            Env=create_env,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "habitat_sim", SimpleNamespace(__version__="0.1.7", __file__="/fixture/habitat_sim/__init__.py")
    )
    monkeypatch.setitem(sys.modules, "habitat_extensions", SimpleNamespace(get_extended_config=lambda path: config))
    monkeypatch.syspath_prepend(str(ROOT))
    backend = Habitat017Backend(ROOT, profile, "configuration")
    backend.create_environment(payload["episodes"][1], payload, dataset, gt, scene, 128)
    assert captured["payload"]["episodes"][0]["episode_id"] == "7"
    assert captured["payload"]["episodes"][0]["scene_id"] == str(scene)
    assert config.TASK.SENSORS == ["INSTRUCTION_SENSOR"]
    assert config.TASK.NDTW.GT_PATH == str(gt)
    assert config.SIMULATOR.RGB_SENSOR.WIDTH == 128


def test_missing_modern_compatibility_code_does_not_fall_back(episode_files, tmp_path):
    profile, dataset, gt, scene, payload = episode_files
    profile["backend"] = "habitat033"
    backend = Habitat033Backend(tmp_path, profile, "configuration")
    with pytest.raises(FileNotFoundError, match="No backend fallback"):
        backend.create_environment(payload["episodes"][0], payload, dataset, gt, scene, 128)


def test_ambiguous_dataset_layout_is_rejected(episode_files):
    profile, dataset, _, _, _ = episode_files
    second = Path(profile["data_root"]) / "R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz"
    second.parent.mkdir(parents=True)
    second.write_bytes(dataset.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        dataset_paths(profile["data_root"], "val_unseen")


def test_runtime_sources_parse_as_python36():
    for path in (ROOT / "vlnce_runtime").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 6))
