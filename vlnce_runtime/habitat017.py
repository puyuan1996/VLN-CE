"""Official Habitat-Lab/Sim 0.1.7 navigation, without importing VLN trainers."""

import importlib
import json
import sys

from vlnce_runtime.episode import NavigationBackend


class Habitat017Backend(NavigationBackend):
    def create_environment(self, episode, payload, dataset_path, gt, scene, resolution):
        habitat = importlib.import_module("habitat")
        simulator = importlib.import_module("habitat_sim")
        for module in (habitat, simulator):
            if getattr(module, "__version__", None) != "0.1.7":
                raise ValueError("habitat017 requires Habitat-Lab and Habitat-Sim version 0.1.7")
        self.versions = {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "habitat_lab": habitat.__version__,
            "habitat_sim": simulator.__version__,
            "habitat_lab_module": habitat.__file__,
            "habitat_sim_module": simulator.__file__,
        }
        sys.path.insert(0, str(self.root))
        extension = importlib.import_module("habitat_extensions")
        filename = "rxr_vlnce_english_task.yaml" if self.dataset == "rxr" else "vlnce_task.yaml"
        config = extension.get_extended_config(str(self.root / "habitat_extensions/config" / filename))
        config.defrost()
        config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
        config.DATASET.SPLIT = self.profile.get("eval_set", "val_unseen")
        config.DATASET.DATA_PATH = str(dataset_path)
        config.DATASET.SCENES_DIR = self.profile["data_root"]
        config.DATASET.EPISODES_ALLOWED = [str(episode["episode_id"])]
        if self.dataset == "rxr":
            config.DATASET.ROLES = [self.selection["role"]]
            config.DATASET.LANGUAGES = list(self.selection["languages"])
        config.TASK.NDTW.SPLIT = self.profile.get("eval_set", "val_unseen")
        config.TASK.NDTW.GT_PATH = str(gt)
        config.TASK.SENSORS = ["INSTRUCTION_SENSOR"]
        config.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.profile["gpu_device_id"]
        for name in ("RGB_SENSOR", "DEPTH_SENSOR"):
            sensor = getattr(config.SIMULATOR, name)
            sensor.WIDTH = resolution
            sensor.HEIGHT = resolution * 3 // 4 if self.dataset == "rxr" else resolution
        config.freeze()
        selected = dict(episode)
        selected["episode_id"] = str(episode["episode_id"])
        selected["scene_id"] = str(scene)
        selected_payload = dict(payload)
        selected_payload["episodes"] = [selected]
        dataset = habitat.make_dataset(config.DATASET.TYPE)
        if self.dataset == "rxr":
            # RxR.from_json requires SPLIT even when loading only one episode.
            # Passing config to the constructor would eagerly load the full split.
            dataset.config = config.DATASET
        dataset.from_json(json.dumps(selected_payload))
        if len(dataset.episodes) != 1 or str(dataset.episodes[0].episode_id) != str(episode["episode_id"]):
            raise ValueError("native dataset did not retain the selected VLN-CE episode")
        return habitat.Env(config=config, dataset=dataset)

    def advance(self, action):
        observations = self.native.step(action)
        return observations, self.native.episode_over, self.metrics()

    def metrics(self):
        return self.native.get_metrics()

    def agent_state(self):
        return self.native.sim.get_agent_state()
