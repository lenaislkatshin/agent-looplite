#!/usr/bin/env bash
# 把同一份录制的分片回放给 Pi TS 和本库，逐项比对。用法：scripts/pi_parity/run.sh <分片文件>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
chunks="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
out="$(mktemp -d)"
port=18765
(cd "$here" && npm install --silent --no-package-lock)
uv run --project "$here/../.." python "$here/server.py" "$chunks" "$port" "$out" &
server=$!
trap 'kill $server; rm -rf "$here/node_modules"' EXIT
sleep 1
(cd "$here" && NO_PROXY='*' no_proxy='*' node pi_replay.mjs "$port" "$out")
(cd "$here/../.." && NO_PROXY='*' no_proxy='*' uv run python scripts/pi_parity/looplite_replay.py "$port" "$out")
uv run --project "$here/../.." python "$here/compare.py" "$out"
echo "结果目录：$out"
