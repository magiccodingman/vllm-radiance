"""Isolated synthetic full-model check of stock-arithmetic speculative GDN."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

RADIANCE_SOURCE = "e4b4a735804c490f52bee2a11735a7a3b29a9c5209bdc0fa34f452f96595aafb"
CONV_SOURCE = "da2d1183c29f68497d0166960c3ca7bedd143b90eead1de3d98e90af0c0f8a4a"
ORACLE_DIRECTORY = "XW6RXQHLHKDBX2FF2LE7HICYZGQCSR5L4HJZ7USZF4L2FZQULJTQ"


def install_hook(
    oracle,
    convolution,
    *,
    capture_layer=None,
    capture_consumed=None,
    stock_convolution=False,
    scan=False,
    indexed=False,
    prefill=False,
    capture_residuals=False,
    stock_norm=False,
    stock_head=False,
    norm_build=None,
    head_group_size=1,
    stock_attention=False,
):
    from qwen_r9700_lab.conformance_instrumentation import HookSet
    from qwen_r9700_lab.conformance_radiance import RadianceProbe

    if hashlib.sha256(convolution.read_bytes()).hexdigest() != CONV_SOURCE:
        raise ValueError("unreviewed convolution candidate")
    if (capture_layer is None) != (capture_consumed is None):
        raise ValueError("transition capture requires both layer and consumed position")
    if capture_layer is not None:
        from capture_gdn_decode_transition import install_transition_hook

        install_transition_hook(
            convolution, layer=capture_layer, consumed=capture_consumed, residuals=capture_residuals
        )
    attach, detach = RadianceProbe.attach, RadianceProbe.detach

    def candidate_attach(self):
        import torch  # isort: skip

        import radiance_gdn as native
        from stock_gdn_recurrent_adapter import StockRecurrentAdapter
        from stock_gdn_sequence import native_sequence

        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != RADIANCE_SOURCE:
            raise ValueError("unreviewed Radiance recurrent-call source")
        model = sys.modules["vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"]
        spec = importlib.util.spec_from_file_location("qwen_stock_gdn_conv_candidate", convolution)
        conv = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = conv
        spec.loader.exec_module(conv)
        repairs = HookSet()
        repairs.replace(model, "causal_conv1d_update", conv.causal_conv1d_update)
        conv_adapter = None
        if stock_convolution:
            from stock_gdn_convolution_adapter import StockConvolutionAdapter

            conv_adapter = StockConvolutionAdapter(torch, conv.causal_conv1d_update)
            repairs.replace(native, "conv_update", conv_adapter.__call__)
        holder = {"adapter": None}
        prefill_adapter = None
        if prefill:
            from stock_gdn_prefill_adapter import StockPrefillAdapter
            from stock_gdn_scan import StockScan

            prefill_adapter = StockPrefillAdapter(
                torch,
                native,
                conv.causal_conv1d_update,
                StockScan(torch, torch.device("cuda", torch.cuda.current_device())),
                native.forward_core_fused,
            )
            repairs.replace(native, "forward_core_fused", prefill_adapter.__call__)

        @functools.wraps(native.recurrent_update)
        def recurrent(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("stock GDN diagnostic does not admit graph capture")
            if holder["adapter"] is None:
                if indexed:
                    from stock_gdn_indexed_adapter import StockIndexedAdapter

                    holder["adapter"] = StockIndexedAdapter(torch)
                    return holder["adapter"](*args, **kwargs)
                runner = native_sequence(
                    oracle / "configs/profiles/gdn-stock-m1-arithmetic-v1.json",
                    oracle
                    / "artifacts/conformance/20260916-gdn-arithmetic-contract-001"
                    / ORACLE_DIRECTORY,
                    oracle / "source/fused_recurrent.py",
                    oracle / "source/fla_op.py",
                )
                if scan:
                    from stock_gdn_scan import StockScan

                    runner = StockScan(torch, runner.device)
                holder["adapter"] = StockRecurrentAdapter(torch, runner)
            return holder["adapter"](*args, **kwargs)

        repairs.replace(native, "recurrent_update", recurrent)
        from stock_target_arithmetic import TargetArithmetic

        target_arithmetic = TargetArithmetic(
            self.runner.model,
            repairs,
            norm=stock_norm,
            head=stock_head,
            norm_build=norm_build,
            head_group_size=head_group_size,
        )
        attention = None
        if stock_attention:
            from stock_m1_attention import StockM1Attention

            attention = StockM1Attention(repairs)
        self._stock_gdn_candidate = (
            repairs,
            holder,
            conv_adapter,
            prefill_adapter,
            target_arithmetic,
            attention,
        )
        attach(self)

    def candidate_detach(self):
        info = self.__dict__.pop("_stock_gdn_candidate", None)
        try:
            return detach(self)
        finally:
            if info is not None:
                repairs, holder, conv_adapter, prefill_adapter, target_arithmetic, attention = info
                repairs.close()
                adapter = holder["adapter"]
                write_private(
                    self.campaign.root / "stock-gdn-candidate.json",
                    seal(
                        {
                            "calls": adapter.calls if adapter is not None else 0,
                            "rows": adapter.rows if adapter is not None else 0,
                            "stock_convolution_calls": conv_adapter.calls if conv_adapter else 0,
                            "single_launch_scan": scan,
                            "indexed_scan": indexed,
                            "prefill_calls": prefill_adapter.calls if prefill_adapter else 0,
                            "prefill_rows": prefill_adapter.rows if prefill_adapter else 0,
                            "target_arithmetic": target_arithmetic.receipt(),
                            "target_attention": attention.receipt() if attention else None,
                            "candidate_sources": {
                                name: hashlib.sha256(
                                    Path(__file__).with_name(name).read_bytes()
                                ).hexdigest()
                                for name in (
                                    "stock_gdn_scan.py",
                                    "stock_gdn_scan_kernel.py",
                                    "stock_gdn_indexed_adapter.py",
                                    "stock_gdn_convolution_adapter.py",
                                    "stock_gdn_prefill_adapter.py",
                                    "stock_target_arithmetic.py",
                                    "stock_m1_attention.py",
                                )
                            },
                            "oracle_root": str(oracle),
                            "hook": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "adapter": hashlib.sha256(
                                Path(__file__)
                                .with_name("stock_gdn_recurrent_adapter.py")
                                .read_bytes()
                            ).hexdigest(),
                            "sequence": hashlib.sha256(
                                (
                                    oracle / "experiments/radiance-public/stock_gdn_sequence.py"
                                ).read_bytes()
                            ).hexdigest(),
                            "convolution": CONV_SOURCE,
                            "radiance_source": RADIANCE_SOURCE,
                            "installed_backend_changed": False,
                        }
                    ),
                )

    RadianceProbe.attach, RadianceProbe.detach = candidate_attach, candidate_detach


def worker(args):
    from qwen_r9700_lab.conformance_boundaries import compare_boundaries
    from qwen_r9700_lab.conformance_scenarios import native, plan_for, tokens
    from qwen_r9700_lab.conformance_state import compare_frames

    spec = private_json(args.spec)
    case = {"context": args.context, "seed": 0}
    forced = tokens(spec, args.accepted + 3, 1701)
    plan = plan_for(spec, case, forced, accepted=[args.accepted, 0])
    serial_plan = plan_for(spec, case, forced)
    paths = [
        args.serial_capture or native(spec, serial_plan, args.output / "serial", speculation=False),
        native(spec, plan, args.output / "d7", speculation=True),
    ]
    schedules = [private_json(path / "schedule.json") for path in paths]
    for schedule, count in zip(schedules, (args.accepted + 3, 3), strict=True):
        authenticate(schedule)
        if len(schedule["frames"]) != count:
            raise ValueError("incomplete stock GDN model comparison")
        if schedule.get("initial_state") != "independent_zero_state":
            raise ValueError("stock model comparison must start from independent initial state")
    serial_frames = {(row["consumed"], row["pending"]): row for row in schedules[0]["frames"]}
    comparisons = []
    for right in schedules[1]["frames"]:
        left = serial_frames.get((right["consumed"], right["pending"]))
        if left is None:
            raise ValueError("M1 replay missed an accepted D7 prefix")
        result = compare_frames(paths[0] / left["name"], paths[1] / right["name"])
        write_private(args.output / (right["name"] + "-comparison.json"), result)
        comparisons.append(
            {
                "consumed": left["consumed"],
                "equal": result["equal"],
                "first_difference": result["first_difference"],
                "sha256": result["sha256"],
            }
        )
    receipts = [private_json(path / "stock-gdn-candidate.json") for path in paths]
    for receipt in receipts:
        authenticate(receipt)
        if receipt["convolution"] != CONV_SOURCE or receipt["radiance_source"] != RADIANCE_SOURCE:
            raise ValueError("reference capture uses a different numerical implementation")
    if receipts[0]["calls"] != 0 or receipts[1]["calls"] <= 0 or receipts[1]["rows"] <= 0:
        raise ValueError("stock recurrent candidate was not dispatched as declared")
    if args.stock_convolution and receipts[1].get("stock_convolution_calls", 0) <= 0:
        raise ValueError("shared stock convolution was not dispatched")
    if args.prefill and any(
        row.get("prefill_calls", 0) <= 0 or row.get("prefill_rows", 0) < args.context
        for row in receipts
    ):
        raise ValueError("stock prefill was not dispatched in both arms")
    for enabled, name in ((args.stock_norm, "norm"), (args.stock_head, "head")):
        if enabled and receipts[1].get("target_arithmetic", {}).get(name + "_calls", 0) <= 0:
            raise ValueError("target " + name + " arithmetic repair was not dispatched")
    if args.stock_attention and receipts[1].get("target_attention", {}).get("calls", 0) <= 0:
        raise ValueError("independent-query attention was not dispatched")
    boundaries = compare_boundaries(
        paths[0] / "boundaries", paths[1] / "boundaries", args.output / "boundary-comparison"
    )
    result = seal(
        {
            "schema": "urn:qwen:stock-gdn-full-model-diagnostic:v1",
            "status": "TESTED"
            if all(x["equal"] for x in comparisons) and boundaries["equal"]
            else "DISCREPANCY",
            "comparisons": comparisons,
            "boundaries": boundaries,
            "dispatch": receipts,
            "initial_prefill_exact": comparisons[0]["equal"],
            "scope": (
                f"Synthetic {args.context}-token prefix, acceptance={args.accepted}, "
                "and two D7 transitions; "
                "stock recurrence and qualified convolution correction; "
                f"shared convolution={args.stock_convolution}, scan={args.scan}, "
                f"indexed={args.indexed}, stock prefill={args.prefill}."
            ),
            "installed_backend_changed": False,
            "serial_capture": str(paths[0]),
        }
    )
    write_private(args.output / "probe-result.json", result)
    print(
        json.dumps({k: result[k] for k in ["status", "initial_prefill_exact", "sha256"]}),
        flush=True,
    )
    return 0 if result["status"] == "TESTED" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "oracle", "convolution", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--capture-layer", type=int)
    parser.add_argument("--capture-consumed", type=int)
    parser.add_argument("--stock-convolution", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--serial-capture", type=Path)
    parser.add_argument("--context", type=int, default=129)
    parser.add_argument("--accepted", type=int, choices=range(8), default=0)
    parser.add_argument("--indexed", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--capture-residuals", action="store_true")
    parser.add_argument("--stock-norm", action="store_true")
    parser.add_argument("--stock-head", action="store_true")
    parser.add_argument("--norm-build", type=Path)
    parser.add_argument("--head-group-size", type=int, choices=(1, 2, 4, 5), default=1)
    parser.add_argument("--stock-attention", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise ValueError("stock GDN model probe requires explicit admission and campaign GPU lease")
    os.umask(0o077)
    if args.prefill and args.serial_capture:
        raise ValueError("changing prefill arithmetic requires a new M1 reference")
    if (args.prefill or args.indexed) and not args.stock_convolution:
        raise ValueError("stock GDN transition requires the shared stock convolution")
    if (args.capture_layer is None) != (args.capture_consumed is None):
        raise ValueError("transition capture requires both layer and consumed position")
    if args.capture_layer is not None:
        from capture_gdn_decode_transition import validate_selection

        validate_selection(args.capture_layer, args.capture_consumed)
    if args.worker:
        return worker(args)
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    args.output.mkdir(mode=0o700)
    hook = args.output / "hook"
    hook.mkdir(mode=0o700)
    (hook / "sitecustomize.py").write_text(
        "import os, traceback\nfrom pathlib import Path\n"
        "try:\n from qualify_stock_gdn_model import install_hook\n"
        f" install_hook(Path({str(args.oracle)!r}), Path({str(args.convolution)!r}), "
        f"capture_layer={args.capture_layer!r}, capture_consumed={args.capture_consumed!r}, "
        f"stock_convolution={args.stock_convolution!r}, scan={args.scan!r}, "
        f"indexed={args.indexed!r}, prefill={args.prefill!r}, "
        f"capture_residuals={args.capture_residuals!r}, stock_norm={args.stock_norm!r}, "
        f"stock_head={args.stock_head!r}, head_group_size={args.head_group_size!r}, "
        f"stock_attention={args.stock_attention!r}, "
        f"norm_build={str(args.norm_build)!r} if {bool(args.norm_build)!r} else None)\n"
        "except BaseException:\n traceback.print_exc()\n os._exit(87)\n"
    )
    spec = private_json(args.spec)
    env = worker_environment(spec, args.output)
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(hook),
            str(Path(__file__).resolve().parent),
            str(args.oracle / "experiments/radiance-public"),
            env["PYTHONPATH"],
        )
    )
    argv = [sys.executable, str(Path(__file__).resolve()), "--worker", "--allow-gpu"]
    for name in ("spec", "oracle", "convolution", "output"):
        argv.extend(["--" + name, str(getattr(args, name))])
    if args.stock_convolution:
        argv.append("--stock-convolution")
    if args.scan:
        argv.append("--scan")
    argv.extend(
        "--" + name
        for name in (
            "indexed",
            "prefill",
            "capture-residuals",
            "stock-norm",
            "stock-head",
            "stock-attention",
        )
        if getattr(args, name.replace("-", "_"))
    )
    argv.extend(["--context", str(args.context)])
    argv.extend(["--accepted", str(args.accepted)])
    argv.extend(["--head-group-size", str(args.head_group_size)])
    if args.norm_build:
        argv.extend(["--norm-build", str(args.norm_build)])
    if args.serial_capture:
        argv.extend(["--serial-capture", str(args.serial_capture)])
    if args.capture_layer is not None:
        argv.extend(
            [
                "--capture-layer",
                str(args.capture_layer),
                "--capture-consumed",
                str(args.capture_consumed),
            ]
        )
    with gpu_lease(args.output / "gpu-lease"):
        with OwnedProcess(argv, args.output / "process", env=env, timeout=3600) as process:
            code = process.wait()
        write_private(
            args.output / "result.json",
            seal(
                {"returncode": code, "status": "COMPLETED" if code == 0 else "FAILED_OR_DISCREPANT"}
            ),
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
