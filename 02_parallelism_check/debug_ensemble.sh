#!/bin/bash
#SBATCH --job-name=debug_ensemble
#SBATCH --account=project_462000675
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=slurmlogs/debug_ensemble_%j.out.log
#SBATCH --error=slurmlogs/debug_ensemble_%j.err.log


# Print job information
echo "=========================================="
echo "LUMI Embedding Ensemble Evaluation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

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
BATCH_SIZE=8  # Reduced for ensemble processing

# List of models to evaluate
MODELS=(
    "intfloat/multilingual-e5-large-instruct"
    "jinaai/jina-embeddings-v3"
    "jinaai/jina-embeddings-v4"
)

# Ensemble configuration
ENSEMBLE_METHODS=("mean" "concat" "max" "weighted_mean")
SIMILARITY_ENSEMBLE=false  # Set to true for similarity-based ensemble
INDIVIDUAL_MODELS=true     # Set to false to skip individual model evaluation
ENSEMBLE_ONLY=false        # Set to true to evaluate ensemble only

# Print configuration
echo ""
echo "Configuration:"
echo "=============="
echo "  Dataset: $DATASET_NAME"
echo "  Language: $LANGUAGE_CODE"
echo "  Split: $SPLIT"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Models: ${MODELS[@]}"
echo "  Ensemble Methods: ${ENSEMBLE_METHODS[@]}"
echo "  Similarity Ensemble: $SIMILARITY_ENSEMBLE"
echo "  Individual Models: $INDIVIDUAL_MODELS"
echo "  Ensemble Only: $ENSEMBLE_ONLY"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "=============="
echo ""

# Function to run evaluation with error handling
run_evaluation() {
    local config_name=$1
    local extra_args=$2
    
    echo "Running evaluation: $config_name"
    echo "Command: python embedding_eval.py --dataset_name \"$DATASET_NAME\" --language_code \"$LANGUAGE_CODE\" --split \"$SPLIT\" --models ${MODELS[@]} --output_dir \"$OUTPUT_DIR\" --batch_size \"$BATCH_SIZE\" --ensemble_methods ${ENSEMBLE_METHODS[@]} $extra_args"
    
    python ensemble.py \
        --dataset_name "$DATASET_NAME" \
        --language_code "$LANGUAGE_CODE" \
        --split "$SPLIT" \
        --models "${MODELS[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --ensemble_methods "${ENSEMBLE_METHODS[@]}" \
        $extra_args
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $config_name completed successfully!"
        return 0
    else
        echo "✗ $config_name failed with exit code: $exit_code"
        return $exit_code
    fi
}

# Run evaluations
echo "Starting ensemble evaluation..."
echo "==============================="

# Determine evaluation strategy
if [ "$ENSEMBLE_ONLY" = true ]; then
    echo "Running ensemble-only evaluation..."
    if [ "$SIMILARITY_ENSEMBLE" = true ]; then
        run_evaluation "Ensemble Only (Similarity)" "--ensemble_only --similarity_ensemble"
    else
        run_evaluation "Ensemble Only (Embedding)" "--ensemble_only"
    fi
elif [ "$INDIVIDUAL_MODELS" = true ]; then
    echo "Running comprehensive evaluation (individual + ensemble)..."
    if [ "$SIMILARITY_ENSEMBLE" = true ]; then
        run_evaluation "Comprehensive (Similarity Ensemble)" "--individual_models --similarity_ensemble"
    else
        run_evaluation "Comprehensive (Embedding Ensemble)" "--individual_models"
    fi
else
    echo "Running individual models only..."
    run_evaluation "Individual Models" "--individual_models"
fi

# Check results
DATASET_BASENAME=$(basename "$DATASET_NAME")
CSV_FILE="$OUTPUT_DIR/${DATASET_BASENAME}.csv"

if [ -f "$CSV_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "EVALUATION RESULTS SUMMARY"
    echo "=========================================="
    
    # Display results with formatting
    echo "Results file: $CSV_FILE"
    echo ""
    
    # Show column headers
    head -1 "$CSV_FILE" | tr ',' '\t'
    echo "----------------------------------------"
    
    # Show results sorted by MRR (descending)
    tail -n +2 "$CSV_FILE" | sort -t',' -k4 -nr | tr ',' '\t'
    
    echo ""
    echo "Best performing model/ensemble:"
    tail -n +2 "$CSV_FILE" | sort -t',' -k4 -nr | head -1 | tr ',' '\t'
    
    echo ""
    echo "=========================================="
    
    # Create summary statistics
    echo "Summary Statistics:"
    echo "==================="
    
    # Count individual vs ensemble models
    individual_count=$(tail -n +2 "$CSV_FILE" | grep -v "ensemble" | wc -l)
    ensemble_count=$(tail -n +2 "$CSV_FILE" | grep "ensemble" | wc -l)
    
    echo "Individual models evaluated: $individual_count"
    echo "Ensemble methods evaluated: $ensemble_count"
    echo "Total evaluations: $((individual_count + ensemble_count))"
    
    # Best MRR scores
    best_individual_mrr=$(tail -n +2 "$CSV_FILE" | grep -v "ensemble" | sort -t',' -k4 -nr | head -1 | cut -d',' -f4)
    best_ensemble_mrr=$(tail -n +2 "$CSV_FILE" | grep "ensemble" | sort -t',' -k4 -nr | head -1 | cut -d',' -f4)
    
    if [ ! -z "$best_individual_mrr" ]; then
        echo "Best individual model MRR: $best_individual_mrr"
    fi
    
    if [ ! -z "$best_ensemble_mrr" ]; then
        echo "Best ensemble MRR: $best_ensemble_mrr"
        
        # Compare ensemble vs individual
        if [ ! -z "$best_individual_mrr" ]; then
            improvement=$(python3 -c "print(f'{(float('$best_ensemble_mrr') - float('$best_individual_mrr')) / float('$best_individual_mrr') * 100:.2f}%')" 2>/dev/null || echo "N/A")
            echo "Ensemble improvement: $improvement"
        fi
    fi
    
else
    echo "❌ No results file found at: $CSV_FILE"
    echo "Evaluation may have failed completely."
    exit 1
fi

# System resource usage summary
echo ""
echo "Resource Usage Summary:"
echo "======================="
echo "Peak memory usage: $(grep VmPeak /proc/$$/status 2>/dev/null || echo "N/A")"
echo "CPU time: $(grep utime /proc/$$/stat 2>/dev/null | awk '{print $14}' || echo "N/A") ticks"

# Cleanup
echo ""
echo "Cleaning up temporary files..."
rm -rf cache/transformers_cache 2>/dev/null
rm -rf __pycache__ 2>/dev/null

echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "End Time: $(date)"
echo "Results saved in: $OUTPUT_DIR"
echo "Log files: logs/embedding_ensemble_${SLURM_JOB_ID}.out"
echo "=========================================="
