"""Oracle collection must translate native actions and retain failed evidence."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vlnce_runtime import demo


@pytest.mark.parametrize("success", [True, False])
def test_native_action_mapping_and_acceptance(tmp_path, monkeypatch, success):
    # Native STOP is 0; the public worker's STOP is 3.
    native_actions = SimpleNamespace(STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3)
    actions_module = ModuleType("habitat.sims.habitat_simulator.actions")
    actions_module.HabitatSimActions = native_actions
    follower_module = ModuleType("habitat.tasks.nav.shortest_path_follower")

    class Follower:
        def __init__(self, sim, goal_radius, return_one_hot):
            self.actions = iter([1, 2, 3, 0])

        def get_next_action(self, goal):
            return next(self.actions)

    follower_module.ShortestPathFollower = Follower
    monkeypatch.setitem(sys.modules, actions_module.__name__, actions_module)
    monkeypatch.setitem(sys.modules, follower_module.__name__, follower_module)
    monkeypatch.setattr(demo, "load_profile", lambda config: ({"backend": "habitat017"}, "hash"))
    calls = []

    class Backend:
        def __init__(self, *args):
            self.native = SimpleNamespace(
                sim=object(), current_episode=SimpleNamespace(goals=[SimpleNamespace(position=[0, 0, 0])])
            )

        def reset(self, params):
            return {"state": {"env_step": 0}}

        def step(self, action):
            calls.append(action)
            return {"state": dict(state_known=True, terminated=action == 3, truncated=False,
                                  task_success=success and action == 3, env_step=len(calls))}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(demo, "Habitat017Backend", Backend)
    before = Path.cwd()
    output = tmp_path / "oracle"
    if success:
        path = demo.collect("runtime.json", output)
        assert json.loads(path.read_text())["actions"] == [0, 1, 2, 3]
    else:
        with pytest.raises(RuntimeError, match="did not complete"):
            demo.collect("runtime.json", output)
        assert not (output / "demo.json").exists()
    assert len(json.loads((output / "transitions.json").read_text())) == 5
    assert calls == [0, 1, 2, 3, "close"]
    assert Path.cwd() == before
