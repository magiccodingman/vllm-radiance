#!/usr/bin/env python3
"""Stop torch's compilation-metrics telemetry from throwing on a non-serializable config value.

torch 2.11 logs a JSON dump of the dynamo and compiler configs after each compile. It filters the
known-unserializable keys through a blocklist, but the blocklist is incomplete -- e.g. it names
`ignore_logger_methods` while the config actually carries `ignore_logging_functions`, a set that is
empty at import and holds a function object by the time a serve compiles. `json.dumps` then raises
`TypeError: Object of type function is not JSON serializable`.

Nothing breaks: torch catches it and the compile completes, but every worker prints three multi-line
tracebacks at startup ("Unexpected exception logging compilation metrics" / "... runtime metrics"),
which reads like a fault during model load and buries real messages.

Fix: render whatever cannot be serialized as its repr instead of raising, at both call sites. The
telemetry keeps working and the offending value still appears in it, just as text. Blocklisting the
one key would leave the next such key to fail the same way.

Idempotent; anchor-count-guarded; ast.parse guard before writing. NOOP once applied."""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "torch/_dynamo/utils.py"
OLD = "    return json.dumps(config_dict, sort_keys=True)"
NEW = ("    # radiance: a function or other non-JSON value in the config must not take down the\n"
       "    # telemetry path; render it as text instead of raising.\n"
       "    return json.dumps(config_dict, sort_keys=True, default=repr)")
SENTINEL = "sort_keys=True, default=repr"


def main():
    # two call sites (dynamo config, compiler config); anchor each with its preceding line so the
    # replacement stays unique and drift is still detected
    apply(
        F,
        "    config_dict = clean_for_json(config.get_config_copy())\n" + OLD,
        "    config_dict = clean_for_json(config.get_config_copy())\n" + NEW,
        SENTINEL,
        "dynamo config metrics: tolerate unserializable values",
    )
    apply(
        F,
        "    config_dict = clean_for_json(compiler_config_copy)\n" + OLD,
        "    config_dict = clean_for_json(compiler_config_copy)\n" + NEW,
        "clean_for_json(compiler_config_copy)\n    # radiance:",
        "compiler config metrics: tolerate unserializable values",
    )


if __name__ == "__main__":
    main()
