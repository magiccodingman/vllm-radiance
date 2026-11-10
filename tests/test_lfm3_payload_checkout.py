"""Exercise Git checkout conversions against the real immutable LFM3 imports."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "upstream/vllm-lfm3"


class LFM3PayloadCheckoutTests(unittest.TestCase):
    def test_actual_import_hashes_survive_checkout_conversion(self):
        manifest = json.loads((VENDOR / "SOURCE_PROVENANCE.json").read_text())
        with tempfile.TemporaryDirectory(prefix="radiance-lfm3-checkout-") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            # Seed with Git blob bytes, not an already converted host checkout.
            for name in manifest["sha256"]:
                blob = subprocess.check_output(["git", "show", f"HEAD:upstream/vllm-lfm3/{name}"], cwd=ROOT)
                self.assertEqual(hashlib.sha256(blob).hexdigest(), manifest["sha256"][name])
                (source / name).write_bytes(blob)
            shutil.copyfile(VENDOR / ".gitattributes", source / ".gitattributes")
            subprocess.run(["git", "-c", "core.autocrlf=false", "add", "."], cwd=source, check=True)
            subprocess.run(["git", "-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid",
                            "commit", "-qm", "immutable imports"], cwd=source, check=True)
            for conversion in ("true", "false", "input"):
                with self.subTest(core_autocrlf=conversion):
                    checkout = Path(directory) / conversion
                    subprocess.run(["git", "-c", f"core.autocrlf={conversion}", "clone", "-q",
                                    "--no-local", str(source), str(checkout)], check=True)
                    for name, expected in manifest["sha256"].items():
                        self.assertEqual(hashlib.sha256((checkout / name).read_bytes()).hexdigest(),
                                         expected, f"{conversion}: {name}")


if __name__ == "__main__":
    unittest.main()
