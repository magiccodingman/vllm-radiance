"""CPU sampling masks, including vocabularies smaller than the candidate cap."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE = Path(os.environ.get(
    "RADIANCE_TOPK_TEST_SOURCE",
    str(Path(__file__).resolve().parents[1] / "radiance_topk.py"),
))
spec = importlib.util.spec_from_file_location("tested_radiance_topk", SOURCE)
topk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topk)


def reference(logits, k, p):
    """Full ascending-sort reference, independent of the candidate window."""
    values, indices = logits.sort(dim=-1)
    threshold = values.gather(1, (logits.shape[-1] - k).unsqueeze(1))
    values = values.masked_fill(values < threshold, float("-inf"))
    if p is not None:
        remove = values.softmax(dim=-1).cumsum(dim=-1) <= 1 - p.unsqueeze(1)
        remove[:, -1] = False
        values = values.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(1, indices, values)


class TopKVocabularyTests(unittest.TestCase):
    def check_mask(self, vocab, cap=64, probability=None):
        generator = torch.Generator(device="cpu").manual_seed(20261018)
        logits = torch.randn((3, vocab), generator=generator, device="cpu")
        k = torch.tensor([1, min(cap, max(1, vocab // 2)), min(cap, max(1, vocab - 1))], device="cpu")
        p = None if probability is None else torch.full((3,), probability, device="cpu")
        expected = reference(logits, k, p)
        with patch.object(topk, "KCAP", cap):
            actual = topk.apply_top_k_top_p_composite(logits, k, p)
        self.assertIs(actual, logits)
        self.assertTrue(torch.equal(actual, expected))

    def test_small_vocab_top_k(self):
        for vocab in (2, 3, 7, 31, 63):
            with self.subTest(vocab=vocab):
                self.check_mask(vocab)

    def test_small_vocab_top_k_top_p(self):
        for vocab in (2, 3, 7, 31, 63):
            for p in (0.7, 0.95, 1.0):
                with self.subTest(vocab=vocab, p=p):
                    self.check_mask(vocab, probability=p)

    def test_single_token_vocabulary(self):
        self.check_mask(1, probability=0.7)

    def test_ordinary_vocabulary_retains_mask(self):
        for vocab in (64, 65, 129, 257):
            with self.subTest(vocab=vocab):
                self.check_mask(vocab, probability=0.95)

    def test_nondefault_cap_larger_than_vocabulary(self):
        self.check_mask(65, cap=128, probability=0.7)

    def test_smaller_nondefault_cap(self):
        self.check_mask(65, cap=8, probability=0.95)

    def test_empty_batch_keeps_shape(self):
        logits = torch.empty((0, 3), device="cpu")
        result = topk.apply_top_k_top_p_composite(logits, torch.empty(0, dtype=torch.long), None)
        self.assertEqual(result.shape, (0, 3))

    def test_small_vocab_remains_fullgraph_traceable(self):
        function = torch.compile(topk.apply_top_k_top_p_composite, backend="eager", fullgraph=True)
        logits = torch.tensor([[0.1, 1.3, -0.7]], device="cpu")
        k = torch.tensor([1], device="cpu")
        self.assertTrue(torch.equal(function(logits.clone(), k, None), reference(logits, k, None)))


if __name__ == "__main__":
    unittest.main()
