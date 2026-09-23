import numpy as np
import pytest

from qwen_r9700_lab import conformance_reference as r
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def test_all_mxfp4_codes_and_signs_in_original_checkpoint_order():
    packed = np.tile(np.arange(256, dtype=np.uint8), (2, 1))
    scales = np.full((2, 16), 127, np.uint8)
    scales[1] = 126
    actual = r.unpack_mxfp4(packed, scales)
    table = np.asarray(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
    )
    expected = np.asarray([table[int(byte) % 16] for byte in range(256) for _ in (0,)], np.float32)
    np.testing.assert_array_equal(actual[0, ::2], expected)
    np.testing.assert_array_equal(actual[0, 1::2], table[np.arange(256) // 16])
    np.testing.assert_array_equal(actual[1], actual[0] / 2)
    assert np.signbit(actual[0, 16])


def test_fp8_all_finite_codes_and_midpoint_ties():
    codes = np.asarray([i for i in range(256) if i not in (127, 255)], np.uint8)
    np.testing.assert_array_equal(r.fp8_encode(r.fp8_decode(codes)), codes)
    values = r.fp8_decode(np.arange(127, dtype=np.uint8))
    mid = (values[:-1] + values[1:]) / 2
    expected = np.arange(126, dtype=np.uint8)
    expected += expected & 1
    np.testing.assert_array_equal(r.fp8_encode(mid), expected)
    assert r.fp8_encode(np.float32(1e10)) == 126
    with pytest.raises(DiagnosticError):
        r.fp8_encode(np.nan)


def test_bf16_rounding_nan_and_nonassociativity():
    bits = np.asarray([0x3F808000, 0x3F818000, 0x7F800001, 0x80000000], np.uint32)
    actual = r.bf16(bits.view(np.float32)).view(np.uint32)
    assert actual[:2].tolist() == [0x3F800000, 0x3F820000]
    assert np.isnan(actual[2:3].view(np.float32)[0])
    assert actual[3] == 0x80000000
    assert r.ordered_sum(np.asarray([2**24, 1, -(2**24)], np.float32)) == 0
    assert r.ordered_sum(np.asarray([2**24, -(2**24), 1], np.float32)) == 1


def test_gdn_has_independent_closed_form_and_extreme_decay():
    q = np.asarray([[1, 0]], np.float32)
    k = np.asarray([[0, 1]], np.float32)
    v = np.asarray([[3, 4]], np.float32)
    old = np.asarray([[[2, 4], [6, 8]]], np.float32)
    out, state = r.gdn_step(
        q, k, v, np.asarray([0], np.float32), np.asarray([0.5], np.float32), old, scale=1
    )
    np.testing.assert_array_equal(state, [[[2, 3.5], [6, 6]]])
    np.testing.assert_array_equal(out, [[2, 6]])
    for gate in (-20, -80, -1000):
        out, state = r.gdn_step(
            q, k, v, np.asarray([gate], np.float32), np.asarray([1], np.float32), old, scale=1
        )
        assert np.isfinite(out).all() and np.isfinite(state).all()
    np.testing.assert_array_equal(state, [[[0, 3], [0, 4]]])
    np.testing.assert_array_equal(old, [[[2, 4], [6, 8]]])


def test_causal_conv_and_attention_hand_computed():
    out, history = r.convolution_step(
        np.asarray([3], np.float32),
        np.asarray([[1, 2]], np.float32),
        np.asarray([[1, 2, 3]], np.float32),
    )
    np.testing.assert_array_equal(history, [[2, 3]])
    np.testing.assert_array_equal(out, r.bf16(r.silu(np.asarray([14], np.float32))))
    keys = np.zeros((2, 1, 2), np.float32)
    vals = np.asarray([[[1, 2]], [[3, 6]]], np.float32)
    result = r.dense_attention(np.ones((2, 2), np.float32), keys, vals)
    np.testing.assert_array_equal(result, [[2, 4], [2, 4]])
