#!/bin/bash
# Run benchmark evaluation on all training runs + baseline

RUNS_DIR="runs"
BENCHMARK_FILE="benchmark.json"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "Error: $BENCHMARK_FILE not found. Run benchmark_data.py first."
    exit 1
fi

# Evaluate sentence-transformer baseline
echo "========================================"
echo "Evaluating baseline"
echo "========================================"
BASELINE_OUT="$RUNS_DIR/baseline_benchmark_results.json"
if [ -f "$BASELINE_OUT" ]; then
    echo "Skipping baseline (already exists: $BASELINE_OUT)"
else
    $PYTHON "$SCRIPT_DIR/benchmark.py" \
        --benchmark-file "$BENCHMARK_FILE" \
        --output-json "$BASELINE_OUT" \
        --no-progress \
        2>&1 | tee "$RUNS_DIR/baseline_benchmark.log"
fi

# Evaluate each training run
for dir in "$RUNS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")

    config="$dir/config.txt"
    if [ ! -f "$config" ]; then
        echo "Skipping $name (no config.txt)"
        continue
    fi

    # Check if already completed
    result_file="$dir/benchmark_results.json"
    if [ -f "$result_file" ]; then
        echo "Skipping $name (already exists: $result_file)"
        continue
    fi

    # Parse config.txt
    layer=$(grep "Layer:" "$config" | head -1 | awk '{print $NF}')
    pooling=$(grep "Pooling:" "$config" | head -1 | awk '{print $NF}')
    mlp_raw=$(grep "MLP head:" "$config" | head -1 | awk '{print $3}')

    layer="${layer:--1}"
    pooling="${pooling:-mean}"

    mlp_flag=""
    if [ "$mlp_raw" = "True" ]; then
        mlp_flag="--mlp-head"
    fi

    # Find latest checkpoint (highest token count)
    checkpoint_dir="$dir/checkpoints"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Skipping $name (no checkpoints/ directory)"
        continue
    fi

    # Prefer embedder.pt (final), otherwise find latest embedder_*k.pt
    if [ -f "$checkpoint_dir/embedder.pt" ]; then
        checkpoint="$checkpoint_dir/embedder.pt"
    else
        checkpoint=$(ls "$checkpoint_dir"/embedder_*k.pt 2>/dev/null \
            | sed 's/.*embedder_\([0-9]*\)k\.pt/\1 &/' \
            | sort -n \
            | tail -1 \
            | awk '{print $2}')
    fi

    if [ -z "$checkpoint" ]; then
        echo "Skipping $name (no checkpoint found)"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Evaluating: $name"
    echo "  checkpoint: $checkpoint"
    echo "  layer=$layer pooling=$pooling mlp=$mlp_raw"
    echo "========================================"

    $PYTHON "$SCRIPT_DIR/benchmark.py" \
        "$checkpoint" \
        --layer "$layer" \
        --pooling "$pooling" \
        $mlp_flag \
        --benchmark-file "$BENCHMARK_FILE" \
        --output-json "$result_file" \
        --no-progress \
        2>&1 | tee "$dir/benchmark.log"
done

echo ""
echo "========================================"
echo "All benchmarks complete."
echo "========================================"

# Print summary table
echo ""
echo "Summary (Avg P@1 across languages):"
echo "----------------------------------------"
for result in "$RUNS_DIR"/*/benchmark_results.json "$RUNS_DIR"/baseline_benchmark_results.json; do
    [ -f "$result" ] || continue
    name=$(basename "$(dirname "$result")")
    if [ "$name" = "runs" ]; then
        name="baseline"
    fi
    avg_p1=$($PYTHON -c "import json; d=json.load(open('$result')); print(f\"{d['average']['avg_p@1']:.1%}\")" 2>/dev/null)
    avg_mrr=$($PYTHON -c "import json; d=json.load(open('$result')); print(f\"{d['average']['avg_mrr']:.3f}\")" 2>/dev/null)
    printf "  %-55s P@1=%s  MRR=%s\n" "$name" "$avg_p1" "$avg_mrr"
done | sort -t= -k2 -rn
