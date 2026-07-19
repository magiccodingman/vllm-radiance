# gfx1201 (R9700 / RDNA4) root fix for amdsmi enumeration.
#
# On this wheel-based ROCm stack, amdsmi_get_processor_handles() only enumerates
# the GPUs if amdsmi initializes before the HIP runtime opens the devices. Once
# HIP initializes (torch/vLLM do this early), amdsmi is locked out and returns an
# empty handle list: IndexError / 0-device cascades across vLLM's ROCm platform
# helpers (platform detection, device_count, get_device_name, gcn-arch, memory).
#
# Verified: initializing amdsmi before HIP enumerates the AMD processors, and the
# handle list survives real HIP context creation, so handle[0] resolves to the
# discrete R9700 (not any iGPU).
# Initializing amdsmi here at interpreter startup, before anything touches HIP, in
# every process (API server, engine, TP workers, model-inspection subprocess),
# keeps amdsmi functional throughout.
#
# Supersedes the per-symptom platform/arch/device-count workarounds.
try:
    import amdsmi

    amdsmi.amdsmi_init()
    # amdsmi_init() alone isn't enough: must enumerate the processors before HIP
    # initializes, else the handle list is empty once HIP opens the devices.
    amdsmi.amdsmi_get_processor_handles()
except Exception:
    # Non-ROCm / amdsmi-less environments: harmless no-op.
    pass
