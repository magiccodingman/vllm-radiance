#!/usr/bin/env python3
"""Make the compressed all-reduce launch geometry env-tunable (RADIANCE_AR_QNB / RADIANCE_AR_QNT).

The wht6 path ships qmax_nb=48 blocks / qnt=1024 threads with the comment "wire-bound; not
sensitive". The accounting says otherwise: per 80 MiB call the kernel moves ~280 MB of LOCAL
DRAM (input read, packed peer+local writes, both packed reads, output write) against ~30 MiB of
wire -- at 550 GB/s that is ~0.45 ms of DRAM work in a 1.32 ms kernel, and 48 workgroups is 1.5
per WGP for those phases. Whether more blocks shorten the DRAM phases (same wire) is a sweep,
not a claim; this patch only makes the sweep possible without an image rebuild.

Idempotent; inert without the env vars (defaults preserved)."""
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "radiance_allreduce.py"
SENTINEL = "RADIANCE_AR_QNB"

OLD = """        self.qnt = 1024        # threads/block for the compressed push (wire-bound; not sensitive)
        self.qmax_nb = 48      # block cap for the compressed path"""
NEW = """        # Threads/block and block cap for the compressed path. The shipped 1024/48 assumed the
        # kernel is wire-bound; its LOCAL traffic (~280 MB/call at the 80 MiB message) says the
        # DRAM phases matter too, so both are sweepable (patch_ar_geometry.py).
        self.qnt = int(os.environ.get("RADIANCE_AR_QNT", "1024"))
        self.qmax_nb = int(os.environ.get("RADIANCE_AR_QNB", "48"))"""

src = TARGET.read_text()
if SENTINEL in src:
    print(f"  NOOP  {TARGET.name} already applied")
    raise SystemExit(0)
if src.count(OLD) != 1:
    print(f"  FAIL  anchor matched {src.count(OLD)}x, expected 1")
    raise SystemExit(1)
TARGET.write_text(src.replace(OLD, NEW, 1))
print(f"  OK    {TARGET.name}")
