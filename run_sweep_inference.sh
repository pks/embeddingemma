#!/bin/bash
# Run inference on all sweep checkpoints using GNU parallel with 2 GPUs

SWEEP_DIR="sweep"

# Generate list of jobs
generate_jobs() {
    for dir in "$SWEEP_DIR"/*/; do
        [ -d "$dir" ] || continue

        name=$(basename "$dir")

        # Parse: {lang}_{pooling}_{head}_layer{layer}_temp{temp}
        pooling=$(echo "$name" | sed 's/^[^_]*_\([^_]*\)_.*/\1/')
        head=$(echo "$name" | sed 's/^[^_]*_[^_]*_\([^_]*\)_.*/\1/')
        layer=$(echo "$name" | sed 's/.*layer\(-[0-9]*\).*/\1/')

        for checkpoint in "$dir"embedder*.pt; do
            [[ "$checkpoint" == *_attn.pt ]] && continue
            [ -f "$checkpoint" ] || continue

            outfile="${checkpoint%.pt}.out"

            # Skip if already complete
            if [ -f "$outfile" ] && grep -q "EXAMPLE_TEXTS Summary" "$outfile"; then
                echo "Skipping (complete): $outfile" >&2
                continue
            fi

            # Output: checkpoint|layer|pooling|mlp_flag|outfile
            mlp_flag=""
            [ "$head" = "mlp" ] && mlp_flag="--mlp-head"
            echo "$checkpoint|$layer|$pooling|$mlp_flag|$outfile"
        done
    done
}

run_inference() {
    local job_slot=$1
    IFS='|' read -r checkpoint layer pooling mlp_flag outfile <<< "$2"
    local gpu=$((job_slot - 1))
    echo "GPU $gpu: $checkpoint"
    #CUDA_VISIBLE_DEVICES=$gpu python inference.py "$checkpoint" --layer "$layer" --pooling "$pooling" $mlp_flag --no-progress > "$outfile" 2>&1
    CUDA_VISIBLE_DEVICES=1 python inference.py "$checkpoint" --layer "$layer" --pooling "$pooling" $mlp_flag --no-progress > "$outfile" 2>&1
}
export -f run_inference

generate_jobs | parallel -j 1 run_inference {%} {}

echo "Done"
