# v0.30 GGZ14 delta and NVFP4 compatibility

Status: CPU and both-R9700 native conversion fixtures PASS; real NVFP4 model
quality/continuous-recompute qualification is deferred because no suitable local
checkpoint was found. This is not broad production NVFP4-format qualification.

## Immutable donor and prior boundary

Donor: https://github.com/GGZ14/vllm-mxfp4 at
`31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`, resolved at mission start.
RX5's A-tiled/NT import (`b08c86244` in Radiance) corresponds to donor
`159f8f6e1e4c0ebab22711178589d525c93977e0`; earlier RX4 audited
`b9d7ecf8bed3d4be5e99ad7582ad0d64bb27fc99` / authored `5950b383`.
The later gated-norm/multisequence GDN fixes already in Radiance are not new
imports. This is a per-component boundary, not a claim that all donor files
were last imported at one commit.

Selected deltas:

| Donor change | Decision |
| --- | --- |
| `20652eccb870aac78c8e026bd535b0acd420c958`, NVFP4 conversion | Adapt E2M1 quantizer; replace broad precision policy and full-partition FP32 allocation |
| `3f542b7`, wider TP1 decode scratch | Import wider N limit and TP-aware allocation; TP2 allocation unchanged |
| `3e093c5` / `496b6b2`, tiled activation followups | Deferred: experimental FP8-stream/A-tiled profile remains off; no qualification claim |
| `b5e12df`, TP3 dummy heads | Not imported; not this TP2 platform contract |
| `d7c1114`, ParoQuant partition rotations | Not imported; new checkpoint format outside this mission |
| `14de9ec`, v0.29 text anchors | Not imported; reconcile directly against exact v0.30 |
| `3f542b7` lazy GDN / `0cadf57` disabling it | Do not import lazy GDN; donor itself records multi-turn corruption |

The converter arithmetic originates with Brian `<brian@localhost.localdomain>`
in GGZ14's `20652ec`. Radiance's changes are policy, bounds, loader integration,
partition correctness, failure handling, provenance and tests. This is selective
adaptation, not a new upstream base or native NVFP4 execution.

## Conversion contract

`RADIANCE_NVFP4_MXFP4=1` opts into conversion of eligible compressed-tensors
NVFP4 linear weights with quantized input only. NVFP4A16 (`input_quant is None`)
fails closed: the native fixture qualified `use_a16=False`, not reinterpretation
of A16 as dynamic FP8 activation quantization. Default-off upstream handling,
unrelated FP8/BF16 layers and ignored `lm_head` remain untouched. Regression
coverage now includes supported quantized-input selection and explicit A16
rejection before backend construction (10 conversion tests total).
`RADIANCE_NVFP4_SOURCE_ID` must identify the original
checkpoint revision. Existing FP8/BF16 layers and `lm_head` retain upstream
representation/selection. The donor's FP8/BF16/LM-head rewrites are not included;
nondefault extra-conversion knobs fail explicitly.

Use original v0.30 parameter creation/loading, including merged logical widths
and separate global divisors. NVFP4 group16 e4m3 scaling becomes MXFP4 group32
e8m0; nibble order is unchanged. Unsupported shapes/scales fail closed. Native
Radiance W4A8 must be available and the resulting layer must be eligible.

CPU arithmetic is bounded by row chunks (`RADIANCE_NVFP4_CHUNK_ROWS`, default128,
maximum4096) and an algorithmic 32MiB temporary estimate (96 bytes/scalar).
This estimate is NOT a measured RSS cap. No complete FP32 partition is created.
Full source packed tensors coexist with one full converted packed/scale output
and native repack storage during installation; these are explicitly not counted
as chunk temporaries. There is no whole-model converted-copy list or cache.
Native repacking occurs on a staged module; failure does not replace source
parameters. Native scratch/runtime allocation can remain allocated after failure,
so model loading must fail, not continue with a partially selected backend.

Exponent choices: `mse` (default), `ocp`, `noclip`. Receipts bind checkpoint,
original metadata, partition divisors, source and converted hashes, code SHA,
donor, policy, layout, chunk size, error and byte accounting. No persisted
conversion artifact is created. Any future artifact cache must verify all these
identities rather than reuse a bare model name.

## Required gates

CPU tests cover partition-specific divisors, deterministic chunk equivalence,
E2M1 ties, zero blocks, rejection, source preservation and precision policy.
Native fixture must cover original installed loader/repack/apply with finite
outputs, deterministic conversion and measured allocator peaks. Requantization
does not imply byte parity with NVFP4: report reconstruction error separately.
No suitable local NVFP4 checkpoint is currently identified; model-level
continuous/recompute, quality, tools and structured-output gates remain required
before broad production format qualification. No large download is authorized
merely to fill this optional gate.

Both gfx1201 devices passed the actual installed compressed-tensors loader,
staged rollback after injected repack failure, and native W4A8 M1/M2/M17/M65
changed-input/deterministic checks. Nontrivial group scales and separate merged
divisors1/2/4 produce relative reconstruction RMS0.111355; native results against
the converted BF16 reference have relative error0.0262–0.0278. These are measured
requantization/activation errors, not a model-logit equivalence tolerance.
The wider TP1 donor shape N34816/K256/M1 also passed native execution.
GPU conversion/install peak increment was37,809,152bytes in the initial native
fixture. That includes native scratch/repack, not CPU arithmetic. The earlier
403,902,464byte process-highwater increment included later reference operations
and must not be reported as conversion-only RSS; the fixture now samples host
highwater immediately after conversion and labels whole-fixture RSS separately.
On the final-image rerun the conversion/install highwater increments were
2,621,440/2,703,360B and whole-fixture peaks1,824,616,448/1,780,584,448B. The
successful conversion follows the injected-failure attempt, so these are warmed
highwater increments, not a cold process RSS bound. The fixture explicitly
constructs the conversion scheme (its default-off environment remains visible
in the receipt); opt-in selection/policy rejection is separately tested in the
CPU/source gate. No default-on conversion is implied.
