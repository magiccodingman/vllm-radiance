# Current-stack PR8 validation

Runtime: published Radiance 1.0.387, vLLM 0.30, PyTorch 2.12 / ROCm 7.14.
The full extras patch applied to libr4d e8de4bc. Original and corrected scan translation units compiled with -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -mcumode.

The original scan reproduces the defect. Corrected 48/16 and 24/8 layouts pass 12 cases each and changed-input graph replay; they ran on separate R9700 devices, not a distributed TP2 model.

The installed upstream DFlash selector passes 200,000 samples per arm at three positions, biased controls, target-only controls and greedy checks.

These are synthetic native contracts and baseline CPU checks. Full-image build, distributed TP2 serving, end-to-end quality and performance promotion remain separate gates.
