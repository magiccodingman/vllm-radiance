# Makefile — reproducibly build the RADIANCE custom HIP kernels for the AMD Radeon AI PRO
# R9700 (gfx1201 / RDNA4).
#
# The kernels are pybind11 + HIP extensions and MUST be compiled with the same ROCm/hipcc they
# load against at runtime — i.e. the toolchain inside the vllm-radiance image. That image ships
# hipcc but NOT make, so by default this Makefile runs on the host and drives hipcc inside a
# throwaway container:
#
#     make                 # build every kernel whose source is present, inside the image
#     make IMAGE=vllm-radiance:dev   # override the image (default tag reads ./VERSION)
#     make radiance_ar_ext.so
#     make verify          # import-check the built .so files inside the image
#     make clean
#
# If you are already in an environment that has hipcc + pybind11 on PATH (e.g. inside the
# container), build them directly instead by overriding RUN:
#
#     make RUN='bash -c'
#
# Note: rocm-bandwidth-test is AMD's upstream tool; the image build compiles it from source (see
# RBT_VERSION in the Dockerfile). It is not built here.

GFX_ARCH ?= gfx1201
IMAGE    ?= vllm-radiance:$(shell cat VERSION 2>/dev/null || echo latest)
PYTHON   ?= python3

# RUN takes a single shell-script string as its last argument. Default: run it inside the image
# with the repo mounted at /work. Override with RUN='bash -c' to build in the current environment.
RUN ?= docker run --rm --entrypoint bash -v "$(CURDIR):/work" -w /work $(IMAGE) -c

BASE_FLAGS = -O3 -std=c++17 -fPIC -shared --offload-arch=$(GFX_ARCH) -Wno-unused-result

# The quantized (fp8) all-reduce sums two dequantized products per element and MUST be built
# with -ffp-contract=off. Otherwise the compiler contracts `a*sa + b*sb` into an FMA that
# absorbs one product with a single rounding; because the "self" and "peer" products are
# swapped between the two tensor-parallel ranks, each rank fuses a different multiply and the
# ranks disagree by ~1 ULP on a few elements — breaking the replicated-state invariant. The
# bf16 all-reduce is a plain sum (no products) and does not need this flag.
radiance_ar_quant_ext.so: EXTRA_FLAGS = -ffp-contract=off

# Kernel extensions with source in this repo (all are baked into the image and enabled by default).
# router_gemm needs -DTEMPORAL, matching how the Dockerfile compiles it.
router_gemm.so: EXTRA_FLAGS = -DTEMPORAL
KERNELS := radiance_ar_ext.so router_gemm.so
ifneq ($(wildcard radiance_ar_quant_ext.hip),)
KERNELS += radiance_ar_quant_ext.so
endif

.DEFAULT_GOAL := all
.PHONY: all verify clean

all: $(KERNELS)
	@echo "[make] built: $(KERNELS)"

# pybind11 + HIP extension -> importable .so. `python3 -m pybind11 --includes` emits the two
# -I flags (python + pybind11 headers) with no shell-quoting headaches.
%.so: %.hip
	@$(RUN) 'set -e; INC=$$($(PYTHON) -m pybind11 --includes); \
	  echo "[hipcc] $@ (arch $(GFX_ARCH))$(if $(EXTRA_FLAGS), $(EXTRA_FLAGS),)"; \
	  hipcc $(BASE_FLAGS) $(EXTRA_FLAGS) $$INC $< -o $@'

# import each built extension to confirm it loads against the image's ROCm/python
verify:
	@$(foreach m,$(basename $(KERNELS)),$(RUN) 'PYTHONPATH=$$PWD $(PYTHON) -c "import $(m)"' \
	  && echo "[verify] import OK: $(m)" || { echo "[verify] FAILED: $(m)"; exit 1; };)

clean:
	rm -f $(KERNELS)
