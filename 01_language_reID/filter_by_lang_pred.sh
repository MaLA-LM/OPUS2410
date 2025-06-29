#!/bin/bash
#SBATCH --job-name=mala-opus-filter-glotlid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

SOURCE_DIR="/scratch/project_462000941/members/zihao/OPUS2410/mala-opus-dedup-2410-LID"
OUTPUT_DIR="/scratch/project_462000941/members/zihao/OPUS2410/mala-opus-dedup-2410-filtered"
NUM_PROC=64
CONF_THRESHOLD=0.0

python ./filter_by_lang_pred.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_proc "$NUM_PROC" \
  --conf_threshold "$CONF_THRESHOLD" \

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"