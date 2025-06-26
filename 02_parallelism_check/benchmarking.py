#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment for LUMI supercomputer"""
    # Set CUDA device if available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logger.info(f"Using CUDA device: {device}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Set environment variables for better performance on LUMI
    os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings

def load_model(model_name):
    """Load sentence transformer model with error handling"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Handle special cases for models that need trust_remote_code
        trust_remote_code = "jina" in model_name.lower()
        
        if trust_remote_code:
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name)
        
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None

def load_dataset(dataset_name, language_code, split):
    """Load dataset with error handling"""
    try:
        from datasets import load_dataset
        
        ds = load_dataset(dataset_name, language_code, split=split, trust_remote_code=True)
        logger.info(f"Successfully loaded dataset: {dataset_name}, language: {language_code}, split: {split}")
        logger.info(f"Dataset size: {len(ds)}")
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name} with language {language_code}: {e}")
        return None

def encode_texts(model, texts, task=None, batch_size=32):
    """Encode texts with batch processing for memory efficiency"""
    try:
        # Handle Jina v4 models with task parameter
        if task and hasattr(model, 'encode'):
            embeddings = model.encode(texts, task=task, batch_size=batch_size, show_progress_bar=True)
        else:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        logger.info(f"Successfully encoded {len(texts)} texts")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to encode texts: {e}")
        return None

def get_correct_translation_ranks(sims):
    """Calculate ranks of correct translations"""
    argsorts = sims.argsort(axis=1)
    correct_translation_ranks = []
    
    for i, argsort in enumerate(argsorts):
        # Find rank of correct translation (diagonal element)
        correct_translation_rank = np.where(argsort[::-1] == i)[0][0] + 1
        correct_translation_ranks.append(correct_translation_rank)
    
    return correct_translation_ranks

def calculate_metrics(ranks):
    """Calculate MRR and average rank"""
    ranks_array = np.array(ranks)
    mrr = (1 / ranks_array).mean()
    avg_rank = ranks_array.mean()
    return mrr, avg_rank

def evaluate_model(model, model_name, source_texts, target_texts, task=None):
    """Evaluate a single model"""
    logger.info(f"Evaluating model: {model_name}")
    
    # Encode texts
    source_emb = encode_texts(model, source_texts, task=task)
    target_emb = encode_texts(model, target_texts, task=task)
    
    if source_emb is None or target_emb is None:
        logger.error(f"Failed to encode texts for model: {model_name}")
        return None, None
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(source_emb, target_emb)
    
    # Get ranks
    ranks = get_correct_translation_ranks(sims)
    
    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ranks)
    
    logger.info(f"{model_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")
    
    return mrr, avg_rank

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence transformer models on translation retrieval")
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="Dataset name (e.g., 'Helsinki-NLP/tatoeba_mt')")
    parser.add_argument("--language_code", type=str, required=True,
                      help="Language code (e.g., 'cha-eng')")
    parser.add_argument("--split", type=str, default="test",
                      help="Dataset split (default: 'test')")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                      help="List of pretrained sentence transformer models to evaluate")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Output directory for results (default: './results')")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for encoding (default: 32)")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset(args.dataset_name, args.language_code, args.split)
    if ds is None:
        logger.error("Failed to load dataset. Exiting.")
        sys.exit(1)
    
    # Get source and target texts
    source_texts = ds["sourceString"]
    target_texts = ds["targetString"]
    
    # Results storage
    results = []
    
    # Evaluate each model
    for model_name in args.models:
        logger.info(f"Loading model: {model_name}")
        model = load_model(model_name)
        
        if model is None:
            logger.warning(f"Skipping model {model_name} due to loading error")
            continue
        
        # Handle special cases for Jina v4 models
        if "jina-embeddings-v4" in model_name:
            # Evaluate with different tasks
            for task in ['retrieval', 'text-matching']:
                task_model_name = f"{model_name}_{task}"
                mrr, avg_rank = evaluate_model(model, task_model_name, source_texts, target_texts, task=task)
                
                if mrr is not None:
                    results.append({
                        'model': task_model_name,
                        'language_code': args.language_code,
                        'split': args.split,
                        'MRR': mrr,
                        'avg_rank': avg_rank
                    })
        else:
            # Regular evaluation
            mrr, avg_rank = evaluate_model(model, model_name, source_texts, target_texts)
            
            if mrr is not None:
                results.append({
                    'model': model_name,
                    'language_code': args.language_code,
                    'split': args.split,
                    'MRR': mrr,
                    'avg_rank': avg_rank
                })
        
        # Clean up model to free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Extract dataset name for filename
        dataset_basename = args.dataset_name.split('/')[-1] if '/' in args.dataset_name else args.dataset_name
        output_file = output_dir / f"{dataset_basename}.csv"
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        
        # Print results summary
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    else:
        logger.error("No results to save. All models failed to evaluate.")
        sys.exit(1)

if __name__ == "__main__":
    main()