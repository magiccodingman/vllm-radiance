"""CPU regression gate for native replay, aliases and private state restoration."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("native_tape_probe", ROOT / "probe_native_d7_replay_tape.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_native_tape_negative_controls():
    probe.run()
