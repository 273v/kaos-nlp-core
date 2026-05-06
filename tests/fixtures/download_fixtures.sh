#!/usr/bin/env bash
# Download Project Gutenberg test fixture files for kaos-nlp-core benchmarks and tests.
#
# Usage: ./tests/fixtures/download_fixtures.sh
#   Run from the kaos-nlp-core directory.

set -euo pipefail

FIXTURE_DIR="$(cd "$(dirname "$0")" && pwd)"

WAR_AND_PEACE_URL="https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
SHAKESPEARE_URL="https://www.gutenberg.org/cache/epub/100/pg100.txt"

download_if_missing() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        echo "✓ ${name} already exists ($(wc -c < "$dest" | tr -d ' ') bytes)"
        return 0
    fi

    echo "Downloading ${name}..."
    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget -q -O "$dest" "$url"
    else
        echo "ERROR: neither curl nor wget found" >&2
        exit 1
    fi
    echo "✓ ${name} downloaded ($(wc -c < "$dest" | tr -d ' ') bytes)"
}

download_if_missing "$WAR_AND_PEACE_URL" "${FIXTURE_DIR}/war_and_peace.txt"
download_if_missing "$SHAKESPEARE_URL"   "${FIXTURE_DIR}/shakespeare.txt"

echo "All fixtures ready."
