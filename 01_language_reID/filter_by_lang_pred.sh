#!/bin/bash
#SBATCH --job-name=mala-opus-filter-glotlid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

NUM_PROC=16
CONF_THRESHOLD=0.9

SOURCE_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
OUTPUT_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID-Threshold-${CONF_THRESHOLD//./_}"

FILELIST="./mala-opus-dedup-2410-ReLID-by-GlotLID-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"

python ./filter_by_lang_pred.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_proc "$NUM_PROC" \
  --conf_threshold "$CONF_THRESHOLD" \
  --filelist "$FILELIST"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
