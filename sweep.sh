#!/bin/bash
# Hyperparameter sweep: layers x temperatures x train languages

OUTPUT_BASE="sweep_results"

# Validation is always Japanese
VAL_PAIRS="jpn_Jpan-eng_Latn eng_Latn-jpn_Jpan"

# Training configurations: name and pairs
TRAIN_CONFIGS=(
    "ja:jpn_Jpan-eng_Latn eng_Latn-jpn_Jpan"
    "de:deu_Latn-eng_Latn eng_Latn-deu_Latn"
)

# Temperatures to try
TEMPS="0.005 0.05 0.5"

# Layers to try (llama3-8b has 32 layers: -1 to -32)
LAYERS="-1 -2 -4 -8 -16 -32"

# Pooling strategies
POOLINGS="mean max last attention"

# MLP head options (empty string = no flag)
MLP_OPTS="linear mlp"

mkdir -p "$OUTPUT_BASE"

for config in "${TRAIN_CONFIGS[@]}"; do
    train_name="${config%%:*}"
    train_pairs="${config#*:}"

    for pooling in $POOLINGS; do
        for mlp in $MLP_OPTS; do
            for temp in $TEMPS; do
                for layer in $LAYERS; do
                    name="${train_name}_${pooling}_${mlp}_layer${layer}_temp${temp}"
                    outdir="$OUTPUT_BASE/$name"

                    if [ -f "$outdir/embedder.pt" ]; then
                        echo "Skipping $name (already exists)"
                        continue
                    fi

                    # Build mlp flag
                    mlp_flag=""
                    if [ "$mlp" = "mlp" ]; then
                        mlp_flag="--mlp-head"
                    fi

                    echo "========================================"
                    echo "Running: train=$train_name pooling=$pooling mlp=$mlp layer=$layer temp=$temp"
                    echo "========================================"

                    # Add run separator to log (with timestamp)
                    echo "" >> "$outdir.log"
                    echo "=== Run started: $(date) ===" >> "$outdir.log"

                    ./train.py \
                        --layer "$layer" \
                        --temperature "$temp" \
                        --pooling "$pooling" \
                        $mlp_flag \
                        --train-pairs $train_pairs \
                        --val-pairs $VAL_PAIRS \
                        --output-dir "$outdir" \
                        -v \
                        | tee -a "$outdir.log"

                    # Run inference with trained checkpoint
                    echo "Running inference..."
                    ./inference.py \
                        "$outdir/embedder.pt" \
                        --layer "$layer" \
                        --pooling "$pooling" \
                        $mlp_flag \
                        | tee -a "$outdir.log"
                done
            done
        done
    done
done

echo ""
echo "========================================"
echo "Sweep complete. Results in $OUTPUT_BASE/"
echo "========================================"

# Print summary of final validation losses
echo ""
echo "Summary (final val loss):"
for log in "$OUTPUT_BASE"/*.log; do
    name=$(basename "$log" .log)
    loss=$(grep "Checkpoint:" "$log" | tail -1 | grep -oP 'val=\K[0-9.]+')
    echo "  $name: $loss"
done | sort -t: -k2 -n
