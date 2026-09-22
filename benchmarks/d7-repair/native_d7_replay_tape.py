"""Replay observed native calls with one substitution and unchanged inputs.

This is an untimed diagnostic. It records the actual compiled/eager callables;
it neither retraces the model nor substitutes a Python mathematical formula for
an observed kernel. A tape is usable only after its complete output reproduces
the ordinary forward. Hidden side effects require an explicit state adapter.
"""

import weakref
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@dataclass(frozen=True)
class Ref:
    index: int


@dataclass
class Call:
    name: str
    function: object
    args: object
    kwargs: object
    result: object
    state: object = None
    cut: object = None


class ReplayValues(dict):
    """Materialize external buffers only when their first consumer executes."""

    def __init__(self, tape):
        super().__init__()
        self.tape = tape
        self.copies = {}

    def __missing__(self, key):
        description = self.tape.external[key]
        if description[0] == "parameter":
            value = description[1]
        elif description[0] == "alias":
            _, source, dtype, shape, strides, offset = description
            origin = self[source]
            value = torch.empty(0, dtype=dtype, device=origin.device).set_(
                origin.untyped_storage(), offset, shape, strides
            )
        else:
            _, storage, dtype, shape, strides, offset = description
            if storage not in self.copies:
                raw, device = self.tape.storages[storage]
                self.copies[storage] = raw.to(device, copy=True)
            raw = self.copies[storage]
            value = torch.empty(0, dtype=dtype, device=raw.device).set_(
                raw.untyped_storage(), offset, shape, strides
            )
        self[key] = value
        return value


class NativeTape:
    """A bounded, alias-preserving tape for one already-prefilled decode group.

    Parameters remain shared and read-only. Other external tensors are cloned
    through their underlying storage, preserving views and in-place aliases.
    A state adapter owns hidden cache writes and must restore authoritative
    state even if the substituted call raises.
    """

    def __init__(
        self,
        parameters=(),
        *,
        state_factory=None,
        context=None,
        byte_limit=2 << 30,
        capture_filter=None,
    ):
        self.parameters = {t.untyped_storage().data_ptr() for t in parameters} - {0}
        self.state_factory = state_factory
        self.context = context or nullcontext
        self.byte_limit = byte_limit
        self.external_bytes = 0
        self.calls = []
        self.refs = {}
        self.next_ref = 0
        self.external = {}
        self.storages = {}
        self.storage_guards = []
        self.storage_origins = {}
        self.busy = False
        self.output = None
        self.qualified = False
        self.states = {}
        self.capture_filter = capture_filter

    def frozen(self, value):
        snapshot = NativeTape(byte_limit=self.byte_limit)
        snapshot.parameters = self.parameters
        return snapshot, snapshot.encode(value)

    @staticmethod
    def thaw(snapshot):
        frame, tree = snapshot
        return frame.decode(tree, ReplayValues(frame))

    def _tree(self, value, encode):
        leaves, spec = tree_flatten(value)
        return ([encode(v) for v in leaves], spec)

    def _external(self, tensor):
        storage = tensor.untyped_storage()
        # GPU addresses are routinely recycled during a forward. Retain only a
        # weak StorageImpl handle, so an old address cannot identify a new
        # allocation while the tensor data itself can still be freed normally.
        key = (str(tensor.device), storage._cdata)
        if storage.data_ptr() in self.parameters:
            return ("parameter", tensor)
        if key in self.storage_origins:
            return (
                "alias",
                self.storage_origins[key],
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
        if key not in self.storages:
            self.storage_guards.append(storage._weak_ref())
            size = storage.nbytes()
            self.external_bytes += size
            if self.external_bytes > self.byte_limit:
                raise DiagnosticError("native tape exceeded its external-storage budget")
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage)
            self.storages[key] = (raw.to("cpu", copy=True), tensor.device)
        return (
            "storage",
            key,
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )

    def close(self):
        for handle in self.storage_guards:
            torch.UntypedStorage._free_weak_ref(handle)
        self.storage_guards.clear()

    def __del__(self):
        self.close()

    def encode(self, value, *, produced=False):
        def leaf(v):
            if not isinstance(v, torch.Tensor):
                return v
            key = id(v)
            previous = self.refs.get(key)
            if previous is None or previous[0]() is not v:
                ref = Ref(self.next_ref)
                self.next_ref += 1
                # Retaining every forward tensor would needlessly pin all
                # layers' intermediate GPU allocations. Weak references also
                # detect a recycled Python id without retaining its storage.
                self.refs[key] = (weakref.ref(v), ref)
                if not produced:
                    self.external[ref.index] = self._external(v)
                storage = v.untyped_storage()
                storage_key = (str(v.device), storage._cdata)
                if storage_key not in self.storage_origins:
                    self.storage_guards.append(storage._weak_ref())
                    self.storage_origins[storage_key] = ref.index
            return self.refs[key][1]

        return self._tree(value, leaf)

    @staticmethod
    def decode(tree, values):
        leaves, spec = tree
        return tree_unflatten([values[v.index] if isinstance(v, Ref) else v for v in leaves], spec)

    @staticmethod
    def bind(tree, result, values):
        actual, structure = tree_flatten(result)
        expected, expected_structure = tree
        if structure != expected_structure or len(actual) != len(expected):
            raise DiagnosticError("native replay changed a call's output structure")
        for key, value in zip(expected, actual, strict=True):
            if isinstance(key, Ref):
                if not isinstance(value, torch.Tensor):
                    raise DiagnosticError("native replay returned a non-tensor output")
                values[key.index] = value
            elif key != value:
                raise DiagnosticError("native replay changed a non-tensor output")

    def invoke(self, name, function, args, kwargs, *, state=None):
        if self.busy:
            return function(*args, **kwargs)
        self.busy = True
        try:
            schema = getattr(function, "_schema", None)
            if schema is not None:
                for i, arg in enumerate(schema.arguments):
                    if arg.alias_info is None or not arg.alias_info.is_write:
                        continue
                    value = args[i] if i < len(args) else kwargs.get(arg.name)
                    leaves, _ = tree_flatten(value)
                    if any(
                        isinstance(v, torch.Tensor)
                        and v.untyped_storage().data_ptr() in self.parameters
                        for v in leaves
                    ):
                        raise DiagnosticError("native tape observed a write to a shared parameter")
            encoded_args, encoded_kwargs = self.encode(args), self.encode(kwargs)
            cut = None
            if self.capture_filter is not None and self.capture_filter(name, args, kwargs):
                cut = [self.frozen((args, kwargs)), None]
            adapter = state
            if adapter is None and self.state_factory is not None:
                adapter = self.state_factory(name, args, kwargs)
            if adapter is not None and id(adapter) not in self.states:
                adapter.capture_before()
                self.states[id(adapter)] = adapter
            result = function(*args, **kwargs)
            if cut is not None:
                cut[1] = self.frozen((args, kwargs, result))
            if adapter is not None:
                adapter.capture_after()
            self.calls.append(
                Call(
                    name,
                    function,
                    encoded_args,
                    encoded_kwargs,
                    self.encode(result, produced=True),
                    adapter,
                    cut,
                )
            )
            return result
        finally:
            self.busy = False

    def finish(self, output):
        self.output = self.encode(output)
        for state in self.states.values():
            state.capture_after()

    def lifetime(self):
        last = {}
        for index, call in enumerate(self.calls):
            for tree in (call.args, call.kwargs, call.result):
                for value in tree[0]:
                    if isinstance(value, Ref):
                        last[value.index] = index
        for value in self.output[0]:
            if isinstance(value, Ref):
                last[value.index] = len(self.calls)
        # Reinterpret views created outside ATen still depend on their dynamic
        # producer. Keep that storage alive through the last alias consumer.
        for key in sorted(last, reverse=True):
            description = self.external.get(key)
            if description is not None and description[0] == "alias":
                source = description[1]
                last[source] = max(last.get(source, -1), last[key])
        release, storage_release = {}, {}
        for key, index in last.items():
            release.setdefault(index, []).append(key)
            description = self.external.get(key)
            if description is not None and description[0] == "storage":
                storage = description[1]
                storage_release[storage] = max(index, storage_release.get(storage, -1))
        return release, storage_release

    def replay(self, substitution=None):
        """Replace at most one observed call; all other calls remain the reference.

        substitution is (event_index, callable). The callable receives exactly
        the reference prefix's arguments. It must obey the same output/alias
        interface; changed private state is retained only within its adapter.
        """
        if self.output is None:
            raise DiagnosticError("native tape has not reached the vocabulary output")
        if substitution is not None and not self.qualified:
            raise DiagnosticError("native tape has no successful reference-output bridge")
        replacement = None if substitution is None else substitution[0]
        if replacement is not None and not 0 <= replacement < len(self.calls):
            raise DiagnosticError("substitution is outside the captured native program")
        values = ReplayValues(self)
        release, storage_release = self.lifetime()
        self.busy = True
        try:
            with self.context(), ExitStack() as states:
                # A changed KV write must remain visible to later attention.
                # Restoring immediately after each call would silently replace
                # candidate cache values with the reference values.
                for state in self.states.values():
                    states.enter_context(state.replay())
                for index, call in enumerate(self.calls):
                    fn = substitution[1] if index == replacement else call.function
                    result = fn(*self.decode(call.args, values), **self.decode(call.kwargs, values))
                    self.bind(call.result, result, values)
                    for key in release.get(index, ()):
                        values.pop(key, None)
                    for storage, end in storage_release.items():
                        if end == index:
                            values.copies.pop(storage, None)
            return self.decode(self.output, values)
        finally:
            self.busy = False

    def validate(self, expected):
        observed = self.replay()
        left, spec = tree_flatten(expected)
        right, other = tree_flatten(observed)
        if spec != other or len(left) != len(right):
            raise DiagnosticError("native tape's reference output structure differs")
        for a, b in zip(left, right, strict=True):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                if a != b:
                    raise DiagnosticError("native tape's reference output differs")
            elif (
                a.dtype != b.dtype
                or a.shape != b.shape
                or not torch.equal(
                    a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
                )
            ):
                raise DiagnosticError(
                    "native tape does not reproduce the complete reference output"
                )
        self.qualified = True
        return observed

    def check_cut(self, index, candidate):
        """Execute one candidate on a frozen correct cut, including private state.

        Exact outputs, mutated arguments and hidden state permit reuse of the
        already validated suffix. A mismatch must be replayed to vocabulary;
        this Boolean is never substituted for a measured top-k mismatch.
        """
        call = self.calls[index]
        if not self.qualified or call.cut is None:
            raise DiagnosticError("native isolated call lacks a validated reference cut")
        args, kwargs = self.thaw(call.cut[0])
        expected = self.thaw(call.cut[1])
        self.busy = True
        try:
            with self.context(), ExitStack() as states:
                if call.state is not None:
                    states.enter_context(call.state.replay())
                result = candidate(*args, **kwargs)
                left, ls = tree_flatten((args, kwargs, result))
                right, rs = tree_flatten(expected)
                if ls != rs:
                    raise DiagnosticError("isolated candidate changed its native interface")
                exact = True
                for a, b in zip(left, right, strict=True):
                    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                        if a.untyped_storage().data_ptr() in self.parameters:
                            if a is not b:
                                raise DiagnosticError("candidate replaced a shared parameter")
                            continue
                        exact &= (
                            a.shape == b.shape
                            and a.dtype == b.dtype
                            and torch.equal(
                                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
                            )
                        )
                    elif isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) or a != b:
                        raise DiagnosticError("candidate changed a non-tensor native argument")
                state_exact = call.state is None or call.state.matches_after()
                return bool(exact and state_exact), bool(exact), bool(state_exact)
        finally:
            self.busy = False


class TapeDispatch(TorchDispatchMode):
    def __init__(self, tape):
        super().__init__()
        self.tape = tape

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.tape.invoke(str(func), func, args, kwargs or {})
