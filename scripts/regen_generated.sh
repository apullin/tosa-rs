#!/usr/bin/env bash
# Regenerate src/generated/tosa_generated.rs from docs/tosa_1.1.fbs.
# TOSA 1.1 only appends ops and attributes to 1.0 (docs/tosa.fbs), so the
# generated code reads and writes both. Pinned to flatc 25.2.10 to match the
# flatbuffers crate; older flatc emits an incompatible size-field API, so this
# refuses any other version.
set -euo pipefail
cd "$(dirname "$0")/.."
FLATC=${FLATC:-flatc}
version=$("$FLATC" --version | awk '{print $NF}')
if [[ "$version" != "25.2.10" ]]; then
  echo "error: need flatc 25.2.10, got $version ($FLATC); set FLATC=/path/to/flatc" >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$FLATC" --rust -o "$tmp" docs/tosa_1.1.fbs
mv "$tmp"/tosa_1.1_generated.rs src/generated/tosa_generated.rs
echo "regenerated src/generated/tosa_generated.rs from docs/tosa_1.1.fbs with flatc $version"
