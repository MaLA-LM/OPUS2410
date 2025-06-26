#!/bin/bash
#SBATCH --job-name=debug_eval
#SBATCH --account=project_462000675
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
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

# Load required modules on LUMI
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false


# Configuration variables - modify these as needed
DATASET_NAME="Helsinki-NLP/tatoeba_mt"
LANGUAGE_CODE="cha-eng"
SPLIT="test"
OUTPUT_DIR="./results"
BATCH_SIZE=16  # Smaller batch size for CPU to manage memory

# List of models to evaluate
MODELS=(
    "intfloat/multilingual-e5-large-instruct"
    "jinaai/jina-embeddings-v3"
    "jinaai/jina-embeddings-v4"
)

# Print configuration
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Language: $LANGUAGE_CODE"
echo "  Split: $SPLIT"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Models: ${MODELS[@]}"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Run the evaluation script
echo "Starting embedding evaluation..."
python benchmarking.py \
    --dataset_name "$DATASET_NAME" \
    --language_code "$LANGUAGE_CODE" \
    --split "$SPLIT" \
    --models "${MODELS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    
    # Display results if CSV file exists
    CSV_FILE="$OUTPUT_DIR/$(basename $DATASET_NAME).csv"
    if [ -f "$CSV_FILE" ]; then
        echo ""
        echo "Results Summary:"
        echo "=================="
        cat "$CSV_FILE"
    fi
else
    echo "Evaluation failed with exit code: $?"
fi

echo "End Time: $(date)"
echo "Job completed."
