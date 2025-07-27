#!/bin/bash
#SBATCH --job-name=embed_eval
#SBATCH --account=project_462000675
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=slurmlogs/eval_%j.out.log
#SBATCH --error=slurmlogs/eval_%j.err.log

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create necessary directories
mkdir -p results
mkdir -p slurmlogs

# Load required modules on LUMI
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
# source /flash/project_462000941/venv/opus2410_env/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false


MODEL="${1:-intfloat/multilingual-e5-large-instruct}"
DATASET_NAME="${2:-Helsinki-NLP/tatoeba_mt}"
SPLIT="${3:-test}"

OUTPUT_DIR="./results"
BATCH_SIZE=16
SKIP_PROCESSED=true

# Print configuration
echo "Configuration:"
echo " Dataset: $DATASET_NAME"
echo " Model: $MODEL"
echo " Split: $SPLIT"
echo " Output Directory: $OUTPUT_DIR"
echo " Batch Size: $BATCH_SIZE"
echo " Skip Processed: $SKIP_PROCESSED"
echo " CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Function to sanitize model name for display
sanitize_model_name() {
    echo "$1" | sed 's/[\/:-]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

# Get sanitized model name for output file prediction
SANITIZED_MODEL=$(sanitize_model_name "$MODEL")
DATASET_BASENAME=$(basename "$DATASET_NAME")
EXPECTED_OUTPUT_FILE="$OUTPUT_DIR/${DATASET_BASENAME}_${SANITIZED_MODEL}.csv"

echo "Expected output file: $EXPECTED_OUTPUT_FILE"
echo ""

# Build command arguments
CMD_ARGS=(
    --dataset_name "$DATASET_NAME"
    --model "$MODEL"
    --split "$SPLIT"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --skip_processed
)

echo "Starting embedding evaluation for model: $MODEL"
echo "Command: python benchmarking.py ${CMD_ARGS[@]}"
echo ""

python benchmarking.py "${CMD_ARGS[@]}"
