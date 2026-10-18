# FP8-stream unsupported-layer fallback

`RADIANCE_FP8_STREAM=1` remains an explicit experiment. Its decoder forward
omits layer scaling, sequence-parallel MoE and expert-gate branches, so the
installer must retain the original forward when any of those branches is live.
Previously both rejection paths logged "left stock" but installed the simplified
forward anyway. The expert-gate boundary also allowed the preceding layer to
defer its down-projection reduction into a layer that rejects the stream.

The installer now preserves the rejected forward and prevents that incoming
partial sum. Supported adjacent layers retain their existing stream contract;
the final layer still reduces its down projection. Production defaults and
compiler/runtime pins are unchanged.

`python -m unittest discover -s tests -p test_fp8_stream_install.py -v`
executes the actual installer functions extracted from the source, with
structural CPU fixtures and distributed-topology stubs. It covers all three
rejection reasons for full and linear attention, the expert-gate boundary,
supported adjacent layers, default-off behavior, TP1 and PP2 rejection.
These checks do not execute Torch, native operators or model inference.

The contributor has no R9700. Candidate-image, GPU and R9700-TP2 qualification
are **NOT RUN**. Before merging or publishing a runtime image, the maintainer
must validate this candidate on dual R9700: startup/capture, supported FP8-stream
layers, an unsupported-layer boundary, and the default-off control. Retain logs
and run the applicable smoke, sampled tool and correctness gates from
`.radiance/tests.yaml`; historical failures remain failures.

This repair does not establish the cause of the compiler `ConstraintViolationError`
reported in [issue #13](https://github.com/magiccodingman/vllm-radiance/issues/13).
That startup failure still needs an exact-image reproduction and dynamic-shape
logs on the affected model; this change must not be presented as its resolution.
