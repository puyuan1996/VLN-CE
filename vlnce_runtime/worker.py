"""Python 3.6-compatible JSONL server; simulator imports happen only at reset."""

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnce_runtime.configuration import PROFILES, load_profile


def serve(configuration, checksum, name="r2r_unseen"):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = None
    try:
        for line in sys.stdin:
            request = {}
            should_close = False
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or type(request.get("id")) is not int:
                    raise ValueError("request must have an integer id")
                method = request.get("method")
                if method == "close":
                    if backend is not None:
                        backend.close()
                    result, should_close = {}, True
                elif method == "reset":
                    if backend is not None:
                        backend.close()
                    profile, verified = load_profile(configuration, name, expected_sha256=checksum)
                    module = importlib.import_module("vlnce_runtime." + profile["backend"])
                    backend_class = getattr(
                        module, "Habitat017Backend" if profile["backend"] == "habitat017" else "Habitat033Backend"
                    )
                    backend = backend_class(Path(__file__).resolve().parents[1], profile, verified)
                    result = backend.reset(request["params"])
                elif method == "step":
                    if backend is None:
                        raise RuntimeError("reset before step")
                    result = backend.step(request["params"]["action"])
                else:
                    raise ValueError("unknown worker method")
                response = {"id": request["id"], "ok": True, "result": result}
                encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                identifier = request.get("id") if isinstance(request, dict) else None
                encoded = json.dumps(
                    {
                        "id": identifier,
                        "ok": False,
                        "error": "environment_unavailable",
                        "message": type(exc).__name__ + ": " + str(exc),
                    }
                )
                should_close = True
            protocol.write(encoded + "\n")
            protocol.flush()
            if should_close:
                break
    finally:
        if backend is not None:
            backend.close()
        protocol.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(PROFILES), default="r2r_unseen")
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()
    return serve(args.runtime_config, args.runtime_sha256, args.env)


if __name__ == "__main__":
    sys.exit(main())
