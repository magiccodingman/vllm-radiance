#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
ATTEMPTS=${ATTEMPTS:-30}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_tool-schema-gate}
RUN_DIR=${RUN_DIR:-${BENCH_ROOT}/runs/${RUN_ID}}
FIXTURE=${FIXTURE:-${BENCH_ROOT}/fixtures/tool-schema-multitool.json}

for command in curl jq sha256sum; do
  command -v "$command" >/dev/null || { echo "Required command missing: $command" >&2; exit 2; }
done
[[ $ATTEMPTS =~ ^[1-9][0-9]*$ ]] || { echo "ATTEMPTS must be a positive integer" >&2; exit 2; }
[[ -f $FIXTURE ]] || { echo "Fixture not found: $FIXTURE" >&2; exit 2; }
mkdir "$RUN_DIR" || { echo "Run directory already exists: $RUN_DIR" >&2; exit 2; }
mkdir "$RUN_DIR/raw"

jq --arg model "$MODEL_NAME" '.model = $model' "$FIXTURE" >"$RUN_DIR/request.json"
sha256sum "$FIXTURE" >"$RUN_DIR/fixture.sha256"
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'base_url=%s\n' "$BASE_URL"
  printf 'model=%s\n' "$MODEL_NAME"
  printf 'attempts=%s\n' "$ATTEMPTS"
} >"$RUN_DIR/manifest.env"

passes=0
failures=0
: >"$RUN_DIR/results.jsonl"
for attempt in $(seq 1 "$ATTEMPTS"); do
  response="$RUN_DIR/raw/$(printf '%03d' "$attempt").json"
  if ! curl -fsS "${BASE_URL%/}/chat/completions" \
      -H 'Content-Type: application/json' \
      --data-binary @"$RUN_DIR/request.json" >"$response"; then
    jq -nc --argjson attempt "$attempt" \
      '{attempt:$attempt,pass:false,error:"HTTP request failed"}' >>"$RUN_DIR/results.jsonl"
    failures=$((failures + 1))
    continue
  fi

  result=$(jq -c --argjson attempt "$attempt" '
    try (
      .choices[0].message.tool_calls[0].function as $function
      | ($function.arguments | fromjson) as $args
      | {
          attempt: $attempt,
          pass: (
            $function.name == "click"
            and ($args | has("reasoning"))
            and ($args | has("confidence"))
            and ($args | has("stepComplete"))
            and ($args.ref == "e12")
          ),
          name: $function.name,
          arguments: $args
        }
    ) catch {attempt:$attempt,pass:false,error:.}
  ' "$response")
  printf '%s\n' "$result" >>"$RUN_DIR/results.jsonl"
  if jq -e '.pass == true' >/dev/null <<<"$result"; then
    passes=$((passes + 1))
  else
    failures=$((failures + 1))
  fi
done

jq -n \
  --arg run_id "$RUN_ID" \
  --arg completed_utc "$(date -u +%FT%TZ)" \
  --argjson attempts "$ATTEMPTS" \
  --argjson passes "$passes" \
  --argjson failures "$failures" \
  '{run_id:$run_id,completed_utc:$completed_utc,attempts:$attempts,passes:$passes,failures:$failures,pass:($failures == 0)}' \
  >"$RUN_DIR/summary.json"

jq . "$RUN_DIR/summary.json"
((failures == 0))
