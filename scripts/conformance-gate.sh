#!/bin/bash
# Conformance test gate script for CI
# Exits with code 1 if conformance tests fail or pass count below floor

set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-./conformance-corpus}"
FLOOR_FILE="${FLOOR_FILE:-./conformance-floor.json}"
REFERENCE_LLAMA_CPP="${REFERENCE_LLAMA_CPP:-}"

echo "=== Running PESTI Conformance Tests ==="
echo "Corpus: $CORPUS_DIR"
echo "Floor file: $FLOOR_FILE"

# Build conformance binary
cargo build --package pesti-conformance --release 2>&1 | tail -5

# Run tests
CONFORMANCE_OUTPUT=$(
    cargo run --quiet --bin conformance -- \
        --corpus "$CORPUS_DIR" \
        $([ -n "$REFERENCE_LLAMA_CPP" ] && echo "--reference-llama-cpp $REFERENCE_LLAMA_CPP") \
        --floor-pass-count 0 \
        --floor-file "$FLOOR_FILE" 2>&1 || true
)

echo "$CONFORMANCE_OUTPUT"

# Extract pass count from output
PASS_COUNT=$(echo "$CONFORMANCE_OUTPUT" | grep -oP 'Conformance complete: (\d+)/' | grep -oP '\d+')

if [ -z "$PASS_COUNT" ]; then
    echo "ERROR: Could not extract pass count from conformance output"
    exit 1
fi

# Check floor file exists and has content
EXPECTED_MIN=0
if [ -f "$FLOOR_FILE" ]; then
    EXPECTED_MIN=$(cat "$FLOOR_FILE" | tr -d '[:space:]')
    echo "Floor threshold: $EXPECTED_MIN (from $FLOOR_FILE)"
fi

# Gate check
if [ "$PASS_COUNT" -lt "$EXPECTED_MIN" ]; then
    echo "FAIL: Pass count ($PASS_COUNT) below floor threshold ($EXPECTED_MIN)"
    exit 1
fi

echo "PASS: Conformance tests OK ($PASS_COUNT models passed)"
exit 0
