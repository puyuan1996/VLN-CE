"""Load only the standalone runtime, without the upstream training package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
