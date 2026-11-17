#!/usr/bin/env python3
"""Install the Radiance MTP policy on the vLLM 0.34 V2 host loop."""
import sysconfig
from pathlib import Path

from _patchlib import apply

GPU = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu"
SPEC = GPU / "spec_decode"


def insert(file, anchor, addition, label):
    apply(file, anchor, addition + anchor, addition.strip(), label)


def replace(file, anchor, replacement, label):
    apply(file, anchor, replacement, replacement.strip(), label)


def main():
    runner = GPU / "model_runner.py"
    insert(runner, "        num_spec_tokens = self.execute_model_state.num_spec_tokens_to_schedule\n",
           "        draft_widths = None\n", "MTP: clear proposal widths")
    anchor = "            with use_workspace_lane(self._draft_workspace_lane):\n                draft_tokens = self.speculator.propose(\n"
    insert(runner, anchor,
           "            from radiance_draft import v2_policy_for\n"
           "            policy = v2_policy_for(self, input_batch, num_spec_tokens)\n"
           "            self.speculator._radiance_policy = policy\n",
           "MTP: initialize proposal policy")
    insert(runner, "            if self.pcp_manager is not None:\n                self.pcp_manager.sync_drafts(\n",
           "            self.speculator._radiance_policy = None\n"
           "            if policy is not None and policy.step:\n"
           "                draft_widths = policy.finish(draft_tokens)\n"
           "                self._radiance_mtp_forwards = policy.step\n"
           "                self._radiance_mtp_widths = draft_widths\n",
           "MTP: assemble controlled proposal")
    replace(runner, "                input_batch, draft_tokens[:, :num_spec_tokens]\n",
            "                input_batch, draft_tokens[:, :num_spec_tokens], draft_widths\n",
            "MTP: publish per-request widths")

    handler = SPEC / "utils.py"
    replace(handler, "        self, input_batch: InputBatch, draft_tokens: torch.Tensor\n",
            "        self, input_batch: InputBatch, draft_tokens: torch.Tensor, draft_widths=None\n",
            "MTP: accept per-request widths")
    replace(handler, "            for req_id in input_batch.req_ids:\n                self._draft_tokens_by_req[req_id] = num_draft_tokens\n",
            "            for i, req_id in enumerate(input_batch.req_ids):\n"
            "                self._draft_tokens_by_req[req_id] = (\n"
            "                    num_draft_tokens if draft_widths is None else draft_widths[i]\n"
            "                )\n", "MTP: record proposal widths")
    replace(handler, "        for req_id, row in zip(input_batch.req_ids, draft_tokens_np):\n            self._draft_tokens_by_req[req_id] = row\n",
            "        for i, (req_id, row) in enumerate(zip(input_batch.req_ids, draft_tokens_np)):\n"
            "            self._draft_tokens_by_req[req_id] = row if draft_widths is None else row[:draft_widths[i]]\n",
            "MTP: respect proposal widths in grammar validation")

    insert(SPEC / "speculator.py", "        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)\n",
           "        policy = getattr(self, \"_radiance_policy\", None)\n"
           "        if policy is not None:\n"
           "            policy.observe(logits, sampled)\n", "MTP: observe confidence")
    for target in ("target_dependent_ar", "autoregressive"):
        replace(SPEC / target / "speculator.py", "            need_eager=is_profile,\n",
                "            need_eager=is_profile or getattr(self, \"_radiance_policy\", None) is not None,\n",
                "MTP: use eager drafter dispatch only during controlled proposals")
    loop = SPEC / "autoregressive/speculator.py"
    replace(loop, "            if self.supports_decode_metadata_update\n            else self._multi_step_decode\n",
            "            if self.supports_decode_metadata_update and getattr(self, \"_radiance_policy\", None) is None\n"
            "            else self._multi_step_decode\n", "MTP: choose the host loop")
    insert(loop, "        for step in range(1, num_speculative_steps):\n            if not skip_attn",
           "        policy = getattr(self, \"_radiance_policy\", None)\n", "MTP: read the proposal policy")
    replace(loop, "        for step in range(1, num_speculative_steps):\n            if not skip_attn",
            "        for step in range(1, num_speculative_steps):\n"
            "            if policy is not None and policy.stop:\n"
            "                break\n"
            "            if not skip_attn", "MTP: terminate the draft loop early")


if __name__ == "__main__":
    main()
