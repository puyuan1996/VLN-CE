"""Select the configured interpreter; never guess or install an environment."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnce_runtime.configuration import load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["r2r_unseen"])
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configuration = (args.runtime_config or root / ".vlnce/runtime.json").expanduser().resolve()
    try:
        profile, checksum = load_profile(configuration, args.env)
        directory = args.work_dir.resolve()
        if not directory.is_dir():
            raise ValueError("work-dir must be an existing episode directory")
        python = str(Path(profile["prefix"]) / "bin/python")
        environment = dict(os.environ)
        environment.update(profile["env"])
        environment["PATH"] = str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
        os.chdir(str(directory))
        os.execve(
            python,
            [
                python,
                "-u",
                str(root / "vlnce_runtime/worker.py"),
                "--runtime-config",
                str(configuration.resolve()),
                "--runtime-sha256",
                checksum,
            ],
            environment,
        )
    except (OSError, ValueError) as exc:
        print("VLN-CE runtime launch failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
