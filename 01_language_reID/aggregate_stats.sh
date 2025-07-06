#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

OUTPUT_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID-Threshold-0_9/partial_stats"

python ./aggregate_stats.py \
    --stats_dir "$OUTPUT_DIR" \
    --output_file "filter_stats_all.tsv"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
