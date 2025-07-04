#!/bin/bash
#SBATCH --job-name=mala-opus-reLID-conlid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

SOURCE_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410"

NUM_PROC=16

# OUTPUT_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
# MODEL_PATH="/scratch/project_462000941/cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin"

OUTPUT_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-ConLID"
MODEL_PATH="/scratch/project_462000941/cache/huggingface/hub/models--epfl-nlp--ConLID/snapshots/59e1e21e2301cb87f1c244bff71579a17eafaa42"

FILELIST="./mala-opus-dedup-2410-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"

python ./re_lang_identify.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_proc "$NUM_PROC" \
  --model_path "$MODEL_PATH" \
  --filelist "$FILELIST"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
