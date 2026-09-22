#!/usr/bin/env python3
"""CPU conversion contracts; optional native backend qualification is separate."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radiance_nvfp4 as nv


class ConversionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(19)
        self.p = torch.randint(0, 256, (11, 32), dtype=torch.uint8, generator=generator)
        self.s = (torch.rand((11, 4), generator=generator) + 0.5).to(torch.float8_e4m3fn)
        self.g = torch.tensor([1.0, 3.0])
        self.kw = dict(source_id="fixture@sha256:test", metadata={"group_size": 16})

    def convert(self, **kw):
        return nv.convert(self.p, self.s, self.g, [5, 6], **self.kw, **kw)

    def test_merged_partitions_keep_distinct_global_scales_and_chunk_invariance(self):
        a, sa, ra = self.convert(chunk_rows=1)
        b, sb, rb = self.convert(chunk_rows=9)
        self.assertTrue(torch.equal(a, b) and torch.equal(sa, sb))
        self.assertEqual(ra["converted_sha256"], rb["converted_sha256"])
        reference = torch.cat((nv.dequant_nvfp4(self.p[:5], self.s[:5], 1),
                               nv.dequant_nvfp4(self.p[5:], self.s[5:], 3)))
        p, s, _ = nv.quant_mxfp4(reference)
        self.assertTrue(torch.equal(a, p) and torch.equal(sa, s))
        reconstructed = nv.dequant_mxfp4(a, sa)
        self.assertTrue(torch.isfinite(reconstructed).all())
        rel = (reconstructed - reference).norm() / reference.norm()
        self.assertLess(float(rel), 0.3)
        self.assertAlmostEqual(float(rel), ra["relative_rms_error"], places=6)

    def test_nonmerged_and_zero_weight(self):
        p, s, receipt = nv.convert(torch.zeros_like(self.p), self.s,
                                  torch.ones(1), [11], **self.kw)
        self.assertEqual(p.count_nonzero(), 0)
        self.assertEqual(receipt["relative_rms_error"], 0)
        self.assertTrue(torch.isfinite(nv.dequant_mxfp4(p, s)).all())

    def test_dequantization_receives_only_bounded_chunks(self):
        sizes = []
        original = nv.dequant_nvfp4
        def observed(p, s, g):
            sizes.append(p.shape[0])
            return original(p, s, g)
        with patch.object(nv, "dequant_nvfp4", observed):
            self.convert(chunk_rows=2)
        self.assertEqual(sum(sizes), 11)
        self.assertLessEqual(max(sizes), 2)

    def test_bad_shapes_metadata_and_divisors_fail(self):
        cases = [dict(packed=self.p.float()), dict(scales=self.s.float()),
                 dict(scales=self.s[:, :3]), dict(widths=[5, 5]),
                 dict(divisors=torch.ones(1)), dict(divisors=torch.tensor([1., 0.])),
                 dict(divisors=torch.tensor([1., float("nan")])),
                 dict(source_id=""), dict(metadata=None), dict(mode="guess"),
                 dict(chunk_rows=0)]
        base = dict(packed=self.p, scales=self.s, divisors=self.g, widths=[5, 6], **self.kw)
        for case in cases:
            with self.subTest(case=case.keys()), self.assertRaises(ValueError):
                nv.convert(**(base | case))

    def test_late_invalid_scale_does_not_mutate_source(self):
        self.s[-1, 0] = float("nan")
        before_p, before_s = self.p.clone(), self.s.view(torch.uint8).clone()
        with self.assertRaises(ValueError):
            self.convert(chunk_rows=1)
        self.assertTrue(torch.equal(before_p, self.p))
        self.assertTrue(torch.equal(before_s, self.s.view(torch.uint8)))

    def test_default_policy_preserves_other_precisions_and_head(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = nv.policy()
            self.assertFalse(cfg["enabled"] or cfg["extra_fp8"] or cfg["extra_bf16"])
            self.assertEqual(cfg["lm_head"], "preserve")
            self.assertIsNone(nv.select_scheme(None, None, "model.proj"))
        with patch.dict(os.environ, {"RADIANCE_NVFP4_MXFP4": "1"}, clear=True):
            self.assertIsNone(nv.select_scheme(None, None, "model.lm_head"))
            with self.assertRaises(ValueError):
                nv.select_scheme(None, None, "model.proj")

    def test_unsupported_extra_conversion_is_explicit(self):
        for key, value in (("RADIANCE_NVFP4_FP8_LAYERS", "mxfp4"),
                           ("RADIANCE_NVFP4_BF16_LAYERS", "in_proj_ba"),
                           ("RADIANCE_NVFP4_LMHEAD", "bf16")):
            with patch.dict(os.environ, {key: value}, clear=True), self.assertRaises(ValueError):
                nv.policy()

    def test_e2m1_ties_use_even_codes(self):
        x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        self.assertEqual(nv.e2m1_index(x).tolist(), [0, 2, 2, 4, 4, 6, 6])


if __name__ == "__main__":
    unittest.main()
