"""Install an isolated official Habitat 0.1.7 runtime and check its required resources.

Run with a modern bootstrap Python; the simulator itself uses Python 3.6.
"""

import argparse
import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "R2R_VLNCE_v1-3": "1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm",
    "R2R_VLNCE_v1-3_preprocessed": "1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr",
}


def run(command, dry=False, **kwargs):
    print("+ " + " ".join(shlex.quote(str(v)) for v in command), flush=True)
    if not dry:
        subprocess.run(list(map(str, command)), check=True, **kwargs)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def resource_check(data_root, task_index):
    sys.path.insert(0, str(ROOT))
    from vlnce_runtime.episode import dataset_paths, scene_path
    data, gt = dataset_paths(str(data_root), "val_unseen")
    episodes = json.loads(gzip.decompress(data.read_bytes()).decode())["episodes"]
    if not 0 <= task_index < len(episodes):
        raise ValueError("task-index outside val_unseen")
    scene = scene_path(episodes[task_index]["scene_id"], str(data_root))
    return {"episode_id": episodes[task_index]["episode_id"], "scene": str(scene),
            "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "reference_path_sha256": hashlib.sha256(gt.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "check"), nargs="?", default="install")
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or shutil.which("conda"))
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=ROOT / ".vlnce/runtime.json")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-assets", action="store_true", help="Use already downloaded official R2R data")
    args = parser.parse_args(argv)
    configuration = args.runtime_config.expanduser().resolve()
    document = json.loads(configuration.read_text()) if configuration.is_file() else {"version": 1, "environments": {}}
    profile = dict(document["environments"].get("r2r_unseen", {}))
    base = configuration.parent
    prefix = (args.prefix or base / profile.get("prefix", "envs/habitat017")).expanduser().resolve()
    data = (args.data_root or base / profile.get("data_root", "data")).expanduser().resolve()
    if profile.get("backend", "habitat017") != "habitat017":
        raise ValueError("Installer targets official habitat017; keep habitat033 in a separate runtime config")
    python = prefix / "bin/python"
    receipt = prefix / ".vlnce-install.json"
    managed = json.loads(receipt.read_text()) if receipt.exists() else None
    environment = dict(os.environ, **profile.get("env", {}))
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = str(prefix / "bin") + os.pathsep + environment.get("PATH", "")
    if args.command == "install" and not args.dry_run:
        profile.update(backend="habitat017", prefix=str(prefix), data_root=str(data))
        profile.setdefault("gpu_device_id", 0)
        profile.setdefault("env", {})
        document["environments"]["r2r_unseen"] = profile
        save(configuration, document)
    if args.command == "install":
        if not python.exists():
            if prefix.exists() and not receipt.exists():
                raise ValueError("Refusing to modify an incomplete unmanaged prefix: " + str(prefix))
            if not args.conda:
                raise ValueError("Provide --conda /path/to/conda")
            run([args.conda, "create", "-y", "-p", prefix, "--override-channels", "-c", "aihabitat",
                 "-c", "conda-forge", "python=3.6", "habitat-sim=0.1.7", "headless", "libopengl", "pip", "cmake<4"], args.dry_run)
            managed = {"installed": False}
            if not args.dry_run:
                save(receipt, managed)
        if managed is not None and not managed["installed"]:
            if not args.dry_run and not (prefix / "lib/libOpenGL.so.0").exists():
                if not args.conda:
                    raise ValueError("Provide --conda /path/to/conda")
                run([args.conda, "install", "-y", "-p", prefix, "--freeze-installed",
                     "--override-channels", "-c", "conda-forge", "libopengl"])
            lab = base / "sources/habitat-lab-0.1.7"
            if not lab.exists():
                run(["git", "clone", "--depth", "1", "--branch", "v0.1.7",
                     "https://github.com/facebookresearch/habitat-lab.git", lab], args.dry_run)
            pip = [python, "-m", "pip", "install"]
            run(pip + ["pip==21.3.1", "setuptools==59.6.0", "wheel==0.37.1", "numpy==1.19.5", "Cython<3"],
                args.dry_run, env=environment)
            run(pip + ["torch==1.10.2+cpu", "torchvision==0.11.3+cpu", "-f",
                       "https://download.pytorch.org/whl/cpu/torch_stable.html"], args.dry_run, env=environment)
            constraints = ROOT / "environments/habitat017-constraints.txt"
            # Extension imports reach baseline utilities (IL data and TensorBoard),
            # but do not require the upstream TensorFlow 1.x training stack.
            run(pip + ["-c", constraints, "-r", lab / "requirements.txt",
                       "lmdb", "webdataset", "tensorboard", "ifcfg", "networkx"],
                args.dry_run, env=environment)
            run(pip + ["--no-deps", "-e", lab], args.dry_run, env=environment)
            run(pip + ["-c", constraints, "dtw==1.4.0", "fastdtw==0.3.4", "gdown==4.4.0"],
                args.dry_run, env=environment)
            if not args.dry_run:
                save(receipt, {"installed": False, "dependencies_installed": True, "habitat_lab_revision": subprocess.check_output(
                    ["git", "-C", str(lab), "rev-parse", "HEAD"], universal_newlines=True).strip()})
        else:
            print("Reusing existing prefix without installing/upgrading packages:", prefix)
    probe = ("import habitat,habitat_sim; assert habitat.__version__=='0.1.7'; "
             "assert habitat_sim.__version__=='0.1.7'; import habitat_extensions; "
             "print('Habitat-Lab/Sim 0.1.7 imports verified')")
    run([python, "-c", probe], args.dry_run, env=dict(environment, PYTHONPATH=str(ROOT)))
    if not args.dry_run:
        if args.command == "install" and receipt.exists():
            receipt_data = json.loads(receipt.read_text())
            receipt_data["installed"] = True
            save(receipt, receipt_data)
    if args.command == "install":
        if not args.skip_assets:
            for name, identifier in ASSETS.items():
                if (data / "datasets" / name).is_dir() or (data / name).is_dir():
                    continue
                archive = data / "downloads" / (name + ".zip")
                if not args.dry_run:
                    archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    run([prefix / "bin/gdown", "https://drive.google.com/uc?id=" + identifier,
                         "--continue", "-O", archive.with_suffix(".zip.part")],
                        args.dry_run, env=environment, timeout=3600)
                    if not args.dry_run:
                        archive.with_suffix(".zip.part").replace(archive)
                if not args.dry_run:
                    destination = data / "datasets"
                    with zipfile.ZipFile(archive) as bundle:
                        for item in bundle.infolist():
                            target = (destination / item.filename).resolve()
                            if os.path.commonpath([str(destination.resolve()), str(target)]) != str(destination.resolve()):
                                raise ValueError("Unsafe dataset archive path")
                        bundle.extractall(destination)
                    save(archive.with_suffix(".source.json"), {"url": identifier,
                         "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    if not args.dry_run:
        print(json.dumps(resource_check(data, args.task_index), indent=2))
    print("Runtime:", configuration)
    print("MP3D scene assets require the official Matterport3D access process; reuse licensed data via --data-root.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("VLN-CE setup failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
