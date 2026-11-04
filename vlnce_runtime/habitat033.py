"""Adapter for the explicitly supplied NavRLEnv Habitat-Sim 0.3.x port."""

import importlib
import os
import sys

from vlnce_runtime.episode import NavigationBackend


class Habitat033Backend(NavigationBackend):
    def create_environment(self, episode, payload, dataset, gt, scene, resolution):
        bridge = self.root / "vlnce_baselines/common/nav_rl_env_habitat033.py"
        if not bridge.is_file():
            raise FileNotFoundError(
                "habitat033 requires a provenance-verified compatibility implementation: "
                + str(bridge)
                + "; the official VLN-CE checkout does not provide it. No backend fallback is performed."
            )
        simulator = importlib.import_module("habitat_sim")
        version = getattr(simulator, "__version__", "")
        if not version.startswith("0.3."):
            raise ValueError("habitat033 requires Habitat-Sim 0.3.x")
        self.versions = {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "habitat_sim": version,
            "habitat_sim_module": simulator.__file__,
        }
        os.environ["HABITAT033_DATASET_SPLIT"] = self.profile.get("eval_set", "val_unseen")
        sys.path.insert(0, str(bridge.parents[1]))
        environment_class = importlib.import_module("common.nav_rl_env_habitat033").NavRLEnv
        config = {
            "SEED": 100,
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": 500},
            "TASK": {
                "SUCCESS_DISTANCE": 3.0,
                "NDTW": {"SPLIT": self.profile.get("eval_set", "val_unseen"), "FDTW": True, "GT_PATH": str(gt), "SUCCESS_DISTANCE": 3.0},
            },
            "SIMULATOR": {
                "SCENE": str(scene),
                "FORWARD_STEP_SIZE": 0.25,
                "TURN_ANGLE": 15.0,
                "TILT_ANGLE": 15.0,
                "HEIGHT": resolution,
                "WIDTH": resolution,
                "HFOV": 90.0,
                "GPU_DEVICE_ID": self.profile["gpu_device_id"],
            },
            "RL": {"SLACK_REWARD": -0.05, "SUCCESS_REWARD": 2.5},
            "DATASET": {
                "TYPE": "VLN-CE-v1",
                "SPLIT": self.profile.get("eval_set", "val_unseen"),
                "DATA_PATH": str(dataset),
                "EPISODES_ALLOWED": [str(episode["episode_id"])],
            },
        }
        selected = dict(episode)
        selected["scene_id"] = str(scene)
        return environment_class(config=config, dataset=[selected])

    def advance(self, action):
        observations, _, done, info = self.native.step(action=action)
        return observations, done, info

    def metrics(self):
        return self.native.get_info({})

    def agent_state(self):
        return self.native.sim.get_agent(0).get_state()
