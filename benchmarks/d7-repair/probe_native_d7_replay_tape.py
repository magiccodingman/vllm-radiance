"""CPU negative controls for the native-call tape; no GPU or model is loaded."""

from contextlib import contextmanager

import torch
from native_d7_replay_tape import NativeTape, TapeDispatch

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def must_fail(fn):
    try:
        fn()
    except DiagnosticError:
        return
    raise AssertionError("negative control was accepted")


def run():
    x = torch.arange(64, dtype=torch.float32)
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    tape = NativeTape([weight])
    with TapeDispatch(tape):
        y = x[:16].clone()
        view = y.view(2, 8)
        view.add_(x[16:32].view(2, 8))
        out = view @ weight
    tape.finish(out)
    must_fail(lambda: tape.replay((0, lambda *a, **k: None)))
    tape.validate(out)
    assert torch.equal(tape.replay(), out)
    assert torch.equal(x, torch.arange(64, dtype=torch.float32))
    must_fail(lambda: tape.validate(out + 1))
    # Inductor can create reinterpret views without an ATen event. Their
    # storage must remain connected to the candidate producer, not a frozen
    # reference copy that would conceal an upstream corruption.
    implicit_view = NativeTape()
    with TapeDispatch(implicit_view):
        hidden = x.clone()
        implicit_view.busy = True
        alias = hidden.view(8, 8)
        implicit_view.busy = False
        view_output = alias + 2
    implicit_view.finish(view_output)
    implicit_view.validate(view_output)
    altered = implicit_view.replay((0, lambda a: a.clone() + 1))
    assert torch.equal(altered, view_output + 1)

    mm = [i for i, call in enumerate(tape.calls) if call.name == "aten.mm.default"]
    assert len(mm) == 1
    bad = tape.replay((mm[0], lambda a, b: a @ b + 1))
    assert not torch.equal(out, bad)
    assert torch.equal(tape.replay(), out)
    protected = NativeTape([weight])
    must_fail(lambda: protected.invoke("mutate", torch.ops.aten.add_.Tensor, (weight, 1), {}))
    # Packed models retain empty placeholder parameters. Their null address
    # must not classify every fresh empty dependency marker as a parameter.
    empty = NativeTape([torch.empty(0)], capture_filter=lambda *a: True)

    def marker(a):
        return a + 1, torch.empty(0)

    result = empty.invoke("marker", marker, (x,), {})
    empty.finish(result)
    empty.validate(result)
    assert empty.check_cut(0, marker) == (True, True, True)

    # External views must preserve a shared storage and offsets on replay.
    alias_tape = NativeTape()
    base = torch.arange(16, dtype=torch.float32)
    a, b = base[:8], base[4:12]
    with TapeDispatch(alias_tape):
        a.add_(1)
        alias_out = b.clone()
    alias_tape.finish(alias_out)
    alias_tape.validate(alias_out)

    class State:
        def __init__(self):
            self.value = 2

        def capture_before(self):
            self.before = self.value

        def capture_after(self):
            self.after = self.value

        def matches_after(self):
            return self.value == self.after

        @contextmanager
        def replay(self):
            self.value = self.before
            try:
                yield
            finally:
                self.value = self.after

    state = State()
    state_tape = NativeTape()

    def advance(v):
        state.value += 1
        return v + state.value

    def changed_write(v):
        state.value = 5
        return v + 3  # Same immediate output, different state for the reader.

    state_out = state_tape.invoke("advance", advance, (x,), {}, state=state)
    state_tape.finish(state_out)
    state_tape.validate(state_out)
    assert state.value == 3
    isolated = NativeTape([weight], capture_filter=lambda *a: True)
    arg = x[:16].reshape(2, 8).clone()
    with TapeDispatch(isolated):
        answer = arg @ weight
    isolated.finish(answer)
    must_fail(lambda: isolated.check_cut(0, isolated.calls[0].function))
    isolated.validate(answer)
    assert isolated.check_cut(0, isolated.calls[0].function) == (True, True, True)
    assert isolated.check_cut(0, lambda a, b: a @ b + 1) == (False, False, True)
    assert torch.equal(arg, x[:16].reshape(2, 8))
    state.value = 2
    isolated_state = NativeTape(capture_filter=lambda *a: True)
    answer = isolated_state.invoke("advance", advance, (x,), {}, state=state)
    isolated_state.finish(answer)
    isolated_state.validate(answer)
    assert isolated_state.check_cut(0, advance) == (True, True, True)
    assert isolated_state.check_cut(0, changed_write) == (False, True, False)
    assert state.value == 3

    def broken(v):
        state.value = -10
        raise RuntimeError("injected native-call failure")

    try:
        state_tape.replay((0, broken))
    except RuntimeError:
        pass
    else:
        raise AssertionError("injected failure was lost")
    assert state.value == 3
    dependent = NativeTape()
    state.value = 2
    first = dependent.invoke("write", advance, (x,), {}, state=state)
    final = dependent.invoke("read", lambda v: v + state.value, (first,), {})
    dependent.finish(final)
    dependent.validate(final)

    changed = dependent.replay((0, changed_write))
    assert torch.equal(changed, final + 2)
    assert state.value == 3
    print(
        "PASS: reference bridge, isolated fault, input aliases, "
        "parameter write rejection, state recovery"
    )


if __name__ == "__main__":
    run()
