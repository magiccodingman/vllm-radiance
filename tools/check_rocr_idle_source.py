#!/usr/bin/env python3
"""Compile actual patched ROCr wait fragments with a deterministic CPU KFD stub.

This checks host-side contracts, not real event delivery or GPU qualification.
The complete runtime build is a separate gate.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def block(source: str, anchor: str) -> str:
    """Keep the upstream body verbatim; fail closed on ambiguous source drift."""
    if source.count(anchor) != 1:
        raise ValueError(f"Expected exactly one source anchor: {anchor}")
    start = source.index(anchor)
    opening = source.index("{", start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return source[start:closing]


def check(source: Path, cxx: str) -> None:
    runtime_root = source / "projects/rocr-runtime/runtime/hsa-runtime"
    runtime = (runtime_root / "core/runtime/runtime.cpp").read_text(encoding="utf-8")
    interrupt = (runtime_root / "core/runtime/interrupt_signal.cpp").read_text(encoding="utf-8")
    lambdas = "\n".join(block(runtime, name) + ";" for name in (
        "auto processEvent =", "auto PrepareInterrupt =", "auto WaitForInterrupt ="))
    notify = block(interrupt, "void InterruptSignal::StoreReleaseAndNotify(")
    polling = block(runtime, "else if (polling && !finish)")
    # Extract the real cap selection, including its dependency on interrupt support.
    caps = re.search(r"const int poll_nap_ceiling_us =\s*.*?;\s*int poll_nap_us = .*?;", runtime, re.S)
    if caps is None:
        raise ValueError("Missing per-batch backoff initialization")
    if runtime.count("->StoreReleaseAndNotify(1)") != 2:
        raise ValueError("Registration/shutdown must both use the notified wake store")
    clear = block(runtime, "void Runtime::AsyncEvents::Clear()")
    if "age_by_event_.clear();" not in clear:
        raise ValueError("Async event teardown must discard cached ages")
    program = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "core/util/poll_backoff.h"
using namespace rocr::core;
using hsa_signal_value_t = int64_t;
struct HsaEvent { uint64_t age; int notifications = 0; };
struct Signal { HsaEvent* event; int releases = 0;
  HsaEvent* EopEvent() { return event; } void Release() { ++releases; }
};
using hsa_signal_handle = Signal*;
int hsaKmtWaitOnMultipleEvents_Ext(HsaEvent** events, uint32_t count, bool, uint32_t, uint64_t* ages) {
  for (uint32_t i = 0; i < count; ++i) ages[i] = events[i]->age;
  return 0;
}
int hsaKmtSetEvent(HsaEvent* event) { ++event->notifications; return 0; }
#define HSAKMT_CALL(call) (call)
struct Runtime { struct Version { bool supports_event_age = true; } version;
  Version KfdVersion() { return version; }
};
namespace atomic { void Store(int64_t* p, int64_t v, std::memory_order) { *p = v; } }
struct InterruptSignal { struct { int64_t value; } signal_{}; HsaEvent* event_ = nullptr;
  void StoreReleaseAndNotify(hsa_signal_value_t value);
};
NOTIFY_BODY
struct Events {
  std::vector<Signal*> signal_;
  std::vector<bool (*)(int64_t, void*)> handler_;
  std::vector<void*> arg_;
  size_t Size() { return signal_.size(); }
  void CopyIndex(size_t i, size_t j) { signal_[i]=signal_[j]; handler_[i]=handler_[j]; arg_[i]=arg_[j]; }
  void PopBack() { signal_.pop_back(); handler_.pop_back(); arg_.pop_back(); }
};
namespace os { std::vector<int> naps; void uSleep(int nap) { naps.push_back(nap); } }
int main() {
  Runtime instance; auto* runtime_singleton_ = &instance;
  std::vector<HsaEvent*> hsa_events;
  std::vector<uint64_t> event_age;
  std::unordered_map<HsaEvent*, uint64_t> event_age_map;
  HsaEvent a{21}, b{75}, fresh{99};
  Signal sa{&a}, sb{&b}, sn{nullptr}, sf{&fresh};
  Events async_events_{{&sa,&sb}, {[](int64_t,void*){return false;}, [](int64_t,void*){return true;}}, {nullptr,nullptr}};
  std::vector<Signal*> signals{&sa,&sb,&sn,&sf};
  auto* hsa_signals = signals.data();
  uint32_t unique_evts = 0;
LAMBDAS
  assert(PrepareInterrupt(0,true)); assert(event_age[0]==1);
  assert(PrepareInterrupt(1,true)); WaitForInterrupt();
  assert(event_age_map.at(&a)==21 && event_age_map.at(&b)==75);
  unique_evts=0; std::swap(signals[0],signals[1]);
  assert(PrepareInterrupt(0,false)); assert(event_age[0]==75);
  assert(PrepareInterrupt(1,false)); assert(event_age[1]==21);
  unique_evts=0; assert(PrepareInterrupt(3,false)); assert(event_age[0]==1);
  instance.version.supports_event_age=false;
  unique_evts=0; assert(PrepareInterrupt(3,true)); assert(event_age[0]==0);
  instance.version.supports_event_age=true;
  unique_evts=0; assert(!PrepareInterrupt(2,false)); assert(unique_evts==0);
  assert(!processEvent(0,0,false)); assert(sa.releases==1);
  assert(!event_age_map.count(&a)); assert(event_age_map.at(&b)==75);
  assert(async_events_.Size()==1 && async_events_.signal_[0]==&sb);
  assert(processEvent(0,0,false)); assert(event_age_map.at(&b)==75);
  signals[0]=&sa; unique_evts=0; assert(PrepareInterrupt(0,false)); assert(event_age[0]==1);
  InterruptSignal wake; wake.event_=&a;
  wake.StoreReleaseAndNotify(1); assert(wake.signal_.value==1 && a.notifications==1);
  wake.event_=nullptr; wake.StoreReleaseAndNotify(2); assert(wake.signal_.value==2 && a.notifications==1);
  for (bool g_use_interrupt_wait : {false,true}) {
    CAPS
    bool polling=true, finish=false, interrupt_wait=false;
    os::naps.clear();
    for (int i=0;i<64;++i) { if (interrupt_wait) {} POLLING }
    assert(!os::naps.empty());
    assert(os::naps.front()==20);
    assert(os::naps.back()==(g_use_interrupt_wait ? 200 : 2000));
    for (int nap : os::naps) assert(nap<=poll_nap_ceiling_us);
    auto count=os::naps.size(); finish=true;
    if (interrupt_wait) {} POLLING
    assert(os::naps.size()==count);
    finish=false; polling=false; interrupt_wait=true;
    if (interrupt_wait) {} POLLING
    assert(os::naps.size()==count);
  }
  std::cout << "ROCr CPU contracts passed: event identity/reorder/removal/reuse, legacy ages, null events, wake notification, mixed/poll-only caps and completed/interrupt guards\n";
}
'''
    for marker, value in [("NOTIFY_BODY", notify), ("LAMBDAS", lambdas), ("CAPS", caps.group()), ("POLLING", polling)]:
        program = program.replace(marker, value)
    with tempfile.TemporaryDirectory(prefix="radiance-rocr-contract-") as directory:
        root = Path(directory)
        path = root / "contracts.cpp"
        path.write_text(program, encoding="utf-8")
        executable = root / "contracts"
        subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-UNDEBUG", "-I", str(runtime_root), str(path), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
        # Revert each repair independently. Every control must build and fail its
        # behavioral assertions, proving these are regression tests, not compile smokes.
        original = subprocess.check_output(["git", "show", "HEAD:projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp"],
                                           cwd=source, text=True, encoding="utf-8")
        controls = {
            "position-keyed-event-age": program.replace(block(runtime, "auto PrepareInterrupt ="), block(original, "auto PrepareInterrupt =")),
            "unthrottled-polling": program.replace("os::uSleep(poll_nap_us);", "/* original polling path has no sleep */"),
            "missing-wake-notification": program.replace("if (event_ != nullptr) HSAKMT_CALL(hsaKmtSetEvent(event_));", "/* original store may omit KFD notification */"),
        }
        for name, control in controls.items():
            if control == program:
                raise ValueError(f"Negative control did not change source: {name}")
            path.write_text(control, encoding="utf-8")
            subprocess.run([cxx, "-std=c++17", "-UNDEBUG", "-I", str(runtime_root), str(path), "-o", str(executable)], check=True)
            result = subprocess.run([str(executable)], capture_output=True)
            if result.returncode == 0:
                raise AssertionError(f"Unfixed control unexpectedly passed: {name}")
            print(f"Negative control rejected: {name}", flush=True)
        # Run the actual five upstream backoff tests; do not rely on the orphaned rocrtst target.
        upstream = source / "projects/rocr-runtime/rocrtst/common/utils_test/poll_backoff_test.cpp"
        executable = root / "upstream-backoff"
        subprocess.run([cxx, "-std=c++17", "-UNDEBUG", "-I", str(runtime_root), str(upstream),
                        "-lgtest_main", "-lgtest", "-pthread", "-o", str(executable)], check=True)
        result = subprocess.run([str(executable)], check=True, text=True, capture_output=True)
        print(result.stdout)
        if "[  PASSED  ] 5 tests." not in result.stdout:
            raise AssertionError("The five upstream backoff tests did not all run and pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check(args.source.resolve(), args.cxx)
