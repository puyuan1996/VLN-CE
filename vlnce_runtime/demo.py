"""Generate one successful native oracle trajectory, outside the policy/worker protocol."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnce_runtime.configuration import load_profile
from vlnce_runtime.habitat017 import Habitat017Backend


def collect(configuration, output, task_index=0, resolution=500, eval_set="val_unseen"):
    profile, checksum = load_profile(configuration, "r2r_train" if eval_set == "train" else "r2r_unseen")
    if profile["backend"] != "habitat017":
        raise ValueError("native oracle collection currently requires the explicit habitat017 backend")
    # Imports stay inside the simulator Python. No oracle tool is exposed to a policy.
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = Habitat017Backend(root, profile, checksum)
    original = Path.cwd()
    actions, transitions = [], []
    try:
        os.chdir(str(output))
        initial = backend.reset(dict(vlnce_path=str(root), eval_set=eval_set,
                                     task_index=task_index, resolution=resolution))
        transitions.append(initial)
        follower = ShortestPathFollower(backend.native.sim, goal_radius=0.5, return_one_hot=False)
        mapping = {getattr(HabitatSimActions, name): index
                   for index, name in enumerate(("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"))}
        goal = backend.native.current_episode.goals[0].position
        for _ in range(500):
            native_action = follower.get_next_action(goal)
            if native_action not in mapping:
                raise RuntimeError("Native shortest-path follower could not produce a legal action")
            action = mapping[native_action]
            transition = backend.step(action)
            actions.append(action)
            transitions.append(transition)
            if transition["state"]["terminated"] or transition["state"]["truncated"]:
                break
        final = transitions[-1]["state"]
        if not (final["state_known"] and final["task_success"] and final["terminated"] and actions[-1] == 3):
            raise RuntimeError("Native oracle did not complete this episode; retained evidence is not an accepted demo")
        result = dict(mode="demo", label="Habitat 原生最短路径专家（oracle）", teacher="Habitat ShortestPathFollower",
                      task_index=task_index, eval_set=eval_set, resolution=resolution,
                      actions=actions, task_success=True, initial=initial["state"], final=final)
        (output / "demo.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return output / "demo.json"
    finally:
        try:
            (output / "transitions.json").write_text(json.dumps(transitions, ensure_ascii=False) + "\n")
        finally:
            try:
                backend.close()
            finally:
                os.chdir(str(original))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-set", choices=("train", "val_unseen"), default="val_unseen")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=500)
    args = parser.parse_args()
    print(collect(args.runtime_config, args.output, args.task_index, args.resolution, args.eval_set))
