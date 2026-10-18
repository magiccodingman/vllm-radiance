# High CPU use while the ROCm engine is idle

For an engine that stays CPU-busy after requests finish, try explicitly setting
`HSA_TOOLS_DISABLE_REGISTER=1` before container startup. This workaround was
reported and independently confirmed in [issue #6](https://github.com/magiccodingman/vllm-radiance/issues/6)
and [issue #12](https://github.com/magiccodingman/vllm-radiance/issues/12).
Those reports concern ROCm 7.14 deployments; they do not establish that every
idle busy loop has the same cause or qualify the current source candidate.

This disables HSA tool/profiler registration. A diagnostic session that needs
those hooks must remove the override and recreate its container. The reporters
retained ordinary vLLM metrics and GPU monitoring. Their measurements are
external evidence, not hardware tests performed by this contributor.

With `docker run`, add `-e HSA_TOOLS_DISABLE_REGISTER=1` among the Docker options
**before the image name**, preserving your existing image and serving arguments.

With this repository's Compose service, save this as an ignored
`docker-compose.override.yml` (merge into an existing override if present):

```yaml
services:
  vllm:
    environment:
      HSA_TOOLS_DISABLE_REGISTER: "1"
```

Then inspect the merged configuration and recreate the service:

```bash
docker compose config
docker compose up -d --force-recreate vllm
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' vllm-radiance
```

Confirm the inspected environment contains `HSA_TOOLS_DISABLE_REGISTER=1`.
Putting the variable only in `.env` does not pass it through unless the Compose
service references it. Restarting an existing container does not update its
environment.

For validation, record the immutable image digest, model, topology and complete
configuration. Run real requests, wait for the running/waiting request metrics
to reach zero, then compare per-thread CPU at idle with the same image and
workload without the override. Check `/health`, inference results, monitoring
and matched throughput after the change. Keep both control and candidate logs.

`HSA_ENABLE_INTERRUPT=1` alone did not fix the reproduction in issue #6.
Profiler event exhaustion remains an unconfirmed explanation there. An engine
that stays busy after this override needs further diagnosis.

The contributor has no R9700: GPU, dual-R9700 TP2 and performance validation are
**NOT RUN**. The maintainer must test the workaround on the exact candidate
image before publishing a runtime recommendation or changing production
defaults. This documentation does not change the image or Compose defaults.
