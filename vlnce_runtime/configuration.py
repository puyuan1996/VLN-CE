"""Explicit runtime configuration, compatible with Python 3.6."""

import hashlib
import json
import os
from pathlib import Path


def load_profile(path, name="r2r_unseen", expected_sha256=None):
    """Validate one profile without importing simulator dependencies."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("runtime configuration changed during launch")
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("runtime.json requires version 1")
    environments = document.get("environments")
    if not isinstance(environments, dict) or name not in environments:
        raise ValueError("runtime.json requires environments." + name)
    profile = environments[name]
    if not isinstance(profile, dict):
        raise ValueError("runtime profile must be an object")
    unknown = set(profile) - {"backend", "prefix", "data_root", "env", "gpu_device_id"}
    if unknown:
        raise ValueError("unknown runtime profile fields: " + ", ".join(sorted(unknown)))
    if profile.get("backend") not in ("habitat017", "habitat033"):
        raise ValueError("backend must explicitly select habitat017 or habitat033")
    if name not in {"r2r_unseen", "r2r_train"}:
        raise ValueError("unknown R2R runtime profile")
    result = dict(profile)
    result["eval_set"] = "train" if name == "r2r_train" else "val_unseen"
    for key in ("prefix", "data_root"):
        value = result.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(key + " must be a nonempty path")
        target = Path(value).expanduser()
        result[key] = str((path.parent / target).resolve())
    python = Path(result["prefix"]) / "bin/python"
    if not python.is_file() or not os.access(str(python), os.X_OK):
        raise ValueError("runtime interpreter missing or not executable: " + str(python))
    if not Path(result["data_root"]).is_dir():
        raise ValueError("data_root is not a directory: " + result["data_root"])
    environment = result.setdefault("env", {})
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not key or "=" in key or "\0" in key or not isinstance(value, str) or "\0" in value
        for key, value in environment.items()
    ):
        raise ValueError("env must map environment variable names to strings")
    device = result.setdefault("gpu_device_id", 0)
    if type(device) is not int or device < 0:
        raise ValueError("gpu_device_id must be a nonnegative integer")
    return result, checksum
