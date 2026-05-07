#!/usr/bin/env bash
# Run the full benchmark suite and save results for tracking over time.
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run and save with auto-generated name
#   ./scripts/run_benchmarks.sh my-branch    # Run and save with custom label
#   ./scripts/run_benchmarks.sh --compare    # Compare latest run against baseline
#
# Results are saved to .benchmarks/ (gitignored).
# Each run produces:
#   .benchmarks/<label>/pytest-benchmark.json   — Python benchmark data
#   .benchmarks/<label>/rust-criterion/         — Criterion HTML reports
#   .benchmarks/<label>/summary.txt             — Human-readable summary

set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-$(date +%Y%m%d-%H%M%S)}"
OUTDIR=".benchmarks/${LABEL}"

if [[ "$LABEL" == "--compare" ]]; then
    # Find the two most recent benchmark dirs
    DIRS=($(ls -1d .benchmarks/*/ 2>/dev/null | sort | tail -2))
    if [[ ${#DIRS[@]} -lt 2 ]]; then
        echo "Need at least 2 benchmark runs to compare. Found: ${#DIRS[@]}"
        exit 1
    fi
    echo "Comparing ${DIRS[0]} vs ${DIRS[1]}"
    # Find the pytest-benchmark save names
    SAVES=($(ls -1 .benchmarks/.benchmarks_cache/ 2>/dev/null | sort | tail -2))
    if [[ ${#SAVES[@]} -ge 2 ]]; then
        uv run pytest tests/bench_*.py \
            --benchmark-only \
            --benchmark-compare="${SAVES[0]},${SAVES[1]}" \
            --benchmark-compare-fail=min:5% \
            2>&1
    else
        echo "No pytest-benchmark saves found for comparison."
    fi
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "kaos-nlp-core benchmark run: ${LABEL}"
echo "============================================================"
echo ""

# ── 1. Ensure fixtures exist ──
echo "--- Checking fixtures ---"
if [[ ! -f tests/fixtures/war_and_peace.txt ]]; then
    echo "Downloading Gutenberg fixtures..."
    ./tests/fixtures/download_fixtures.sh
fi
if [[ ! -f tests/fixtures/usc.jsonl ]]; then
    echo "Downloading HuggingFace fixtures..."
    uv run --with huggingface_hub,tokenizers,datasets python tests/fixtures/download_hf_fixtures.py
fi
echo "Fixtures ready."
echo ""

# ── 2. Build release ──
echo "--- Building release ---"
uv run maturin develop --release 2>&1 | tail -2
echo ""

# ── 3. Run Rust tests ──
echo "--- Rust tests ---"
cargo test --no-default-features 2>&1 | tail -3
echo ""

# ── 4. Run Python tests ──
echo "--- Python tests ---"
uv run pytest tests/test_*.py --tb=short -q 2>&1 | tail -3
echo ""

# ── 5. Run Python benchmarks ──
echo "--- Python benchmarks ---"
uv run pytest tests/bench_*.py \
    --benchmark-only \
    --benchmark-disable-gc \
    --benchmark-save="${LABEL}" \
    --benchmark-json="${OUTDIR}/pytest-benchmark.json" \
    2>&1 | tee "${OUTDIR}/pytest-benchmark-output.txt"
echo ""

# ── 6. Run Rust criterion benchmarks ──
echo "--- Rust criterion benchmarks ---"
cargo bench --no-default-features 2>&1 | tee "${OUTDIR}/criterion-output.txt"
# Copy criterion reports
if [[ -d target/criterion/report ]]; then
    cp -r target/criterion/report "${OUTDIR}/criterion-report"
fi
echo ""

# ── 7. Generate summary ──
echo "--- Generating summary ---"
uv run python3 << 'PYEOF' > "${OUTDIR}/summary.txt"
import json, sys, os, time
from datetime import datetime

label = os.environ.get("LABEL", "unknown")
print(f"kaos-nlp-core Benchmark Summary")
print(f"Run: {label}")
print(f"Date: {datetime.now().isoformat()}")
print(f"System: {os.uname().sysname} {os.uname().machine}")
print()

# Parse pytest-benchmark JSON
outdir = os.environ.get("OUTDIR", ".")
json_path = f"{outdir}/pytest-benchmark.json"
if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)

    print(f"Python benchmarks: {len(data['benchmarks'])} tests")
    print()

    # Group by group
    groups = {}
    for b in data["benchmarks"]:
        group = b.get("group", "ungrouped")
        groups.setdefault(group, []).append(b)

    for group_name, benchmarks in sorted(groups.items()):
        print(f"  [{group_name}]")
        for b in sorted(benchmarks, key=lambda x: x["stats"]["median"]):
            name = b["name"].replace("test_", "")
            median_us = b["stats"]["median"] * 1e6
            p95_us = b["stats"]["iqr"] * 1e6  # approximate
            ops = b["stats"]["ops"]
            print(f"    {name:<50} {median_us:>10.1f} µs  ({ops:>10.1f} ops/s)")
        print()
else:
    print("  (no pytest-benchmark JSON found)")
PYEOF

LABEL="${LABEL}" OUTDIR="${OUTDIR}" uv run python3 -c "pass"  # ensure env vars work
cat "${OUTDIR}/summary.txt"

echo ""
echo "Results saved to: ${OUTDIR}/"
echo "  pytest-benchmark.json  — raw benchmark data"
echo "  summary.txt            — human-readable summary"
echo "  criterion-report/      — Rust benchmark HTML reports"
echo ""
echo "To compare against this baseline later:"
echo "  ./scripts/run_benchmarks.sh --compare"
