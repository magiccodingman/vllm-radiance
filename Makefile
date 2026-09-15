# Development helper for the pinned R4D kernel library.
#
# R4D lives in a separate repository. The production image fetches an immutable
# tag+commit pair; this helper intentionally uses the same pins and compiles with
# the image's ROCm/Python toolchain:
#
#     make r4d
#     make verify
#     make clean
#
# Maintenance/control-plane helpers:
#
#     make validate
#     make test
#     make test-changed
#     make qualify-upgrade
#     make qualify-release

GFX_ARCH ?= gfx1201
IMAGE    ?= vllm-radiance:$(shell cat VERSION 2>/dev/null || echo latest)
PYTHON   ?= python3
R4D_REPO    ?= $(shell sed -n 's/^ARG R4D_REPO=//p' Dockerfile)
R4D_VERSION ?= $(shell sed -n 's/^ARG R4D_VERSION=//p' Dockerfile)
R4D_COMMIT  ?= $(shell sed -n 's/^ARG R4D_COMMIT=//p' Dockerfile)
R4D_DIR     ?= libr4d

RUN = docker run --rm --entrypoint bash -v "$(CURDIR)/$(R4D_DIR):/work" -w /work $(IMAGE) -c

.DEFAULT_GOAL := r4d
.PHONY: r4d verify clean validate test test-pr test-changed qualify-upgrade qualify-release

$(R4D_DIR):
	git clone --filter=blob:none $(R4D_REPO) $(R4D_DIR)

r4d: $(R4D_DIR)
	@git -C $(R4D_DIR) fetch --depth 1 origin tag $(R4D_VERSION)
	@git -C $(R4D_DIR) checkout -q $(R4D_COMMIT)
	@test "$$(git -C $(R4D_DIR) rev-parse HEAD)" = "$(R4D_COMMIT)" || { \
	  echo "R4D checkout does not match R4D_COMMIT=$(R4D_COMMIT)" >&2; exit 1; }
	@$(RUN) 'GFX_ARCH=$(GFX_ARCH) PYTHON=$(PYTHON) ./build.sh'
	@echo "[make] built $(R4D_DIR)/r4d.so from $(R4D_VERSION) ($(R4D_COMMIT))"

verify:
	@$(RUN) 'PYTHONPATH=$$PWD $(PYTHON) -c "import torch, r4d; \
	  assert r4d.__version__ == \"$(patsubst v%,%,$(R4D_VERSION))\"; \
	  print(\"[verify] r4d\", r4d.__version__, \"OK:\", [n for n in dir(r4d) if not n.startswith(\"_\")])"'

clean:
	rm -rf -- $(R4D_DIR)

validate:
	$(PYTHON) tools/validate_control_plane.py

test: test-pr

test-pr:
	$(PYTHON) tools/radiance.py test pr

test-changed:
	$(PYTHON) tools/radiance.py test changed

qualify-upgrade:
	$(PYTHON) tools/radiance.py qualify upgrade

qualify-release:
	$(PYTHON) tools/radiance.py qualify release
