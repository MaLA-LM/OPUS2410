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
    """Evaluate a single model and return embeddings and metrics"""
    logger.info(f"Evaluating model: {model_name}")
    
    # Encode texts
    source_emb = encode_texts(model, source_texts, task=task)
    target_emb = encode_texts(model, target_texts, task=task)
    
    if source_emb is None or target_emb is None:
        logger.error(f"Failed to encode texts for model: {model_name}")
        return None, None, None, None
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(source_emb, target_emb)
    
    # Get ranks
    ranks = get_correct_translation_ranks(sims)
    
    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ranks)
    
    logger.info(f"{model_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")
    
    return mrr, avg_rank, source_emb, target_emb

def evaluate_ensemble(embeddings_list, model_names, aggregation_method='mean'):
    """Evaluate ensemble of models by aggregating their embeddings"""
    logger.info(f"Evaluating ensemble with {len(embeddings_list)} models using {aggregation_method} aggregation")
    
    if not embeddings_list:
        logger.error("No embeddings provided for ensemble evaluation")
        return None, None
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Aggregate embeddings
    if aggregation_method == 'mean':
        # Average embeddings across models
        ensemble_source_emb = np.mean([emb[0] for emb in embeddings_list], axis=0)
        ensemble_target_emb = np.mean([emb[1] for emb in embeddings_list], axis=0)
    elif aggregation_method == 'concat':
        # Concatenate embeddings
        ensemble_source_emb = np.concatenate([emb[0] for emb in embeddings_list], axis=1)
        ensemble_target_emb = np.concatenate([emb[1] for emb in embeddings_list], axis=1)
    elif aggregation_method == 'max':
        # Element-wise maximum
        ensemble_source_emb = np.maximum.reduce([emb[0] for emb in embeddings_list])
        ensemble_target_emb = np.maximum.reduce([emb[1] for emb in embeddings_list])
    else:
        logger.error(f"Unknown aggregation method: {aggregation_method}")
        return None, None
    
    # Calculate similarities for ensemble
    ensemble_sims = cosine_similarity(ensemble_source_emb, ensemble_target_emb)
    
    # Get ranks
    ensemble_ranks = get_correct_translation_ranks(ensemble_sims)
    
    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ensemble_ranks)
    
    ensemble_name = f"ensemble_{aggregation_method}({'+'.join(model_names)})"
    logger.info(f"{ensemble_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")
    
    return mrr, avg_rank

def aggregate_similarities(similarities_list, model_names, aggregation_method='mean'):
    """Alternative approach: aggregate similarity matrices directly"""
    logger.info(f"Aggregating similarities from {len(similarities_list)} models using {aggregation_method}")
    
    if not similarities_list:
        logger.error("No similarities provided for aggregation")
        return None, None
    
    # Aggregate similarity matrices
    if aggregation_method == 'mean':
        ensemble_sims = np.mean(similarities_list, axis=0)
    elif aggregation_method == 'max':
        ensemble_sims = np.maximum.reduce(similarities_list)
    elif aggregation_method == 'min':
        ensemble_sims = np.minimum.reduce(similarities_list)
    elif aggregation_method == 'weighted_mean':
        # Simple equal weighting for now, can be extended to learned weights
        weights = np.ones(len(similarities_list)) / len(similarities_list)
        ensemble_sims = np.average(similarities_list, axis=0, weights=weights)
    else:
        logger.error(f"Unknown similarity aggregation method: {aggregation_method}")
        return None, None
    
    # Get ranks
    ensemble_ranks = get_correct_translation_ranks(ensemble_sims)
    
    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ensemble_ranks)
    
    ensemble_name = f"ensemble_sim_{aggregation_method}({'+'.join(model_names)})"
    logger.info(f"{ensemble_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")
    
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
    parser.add_argument("--ensemble_methods", type=str, nargs="+", 
                      default=["mean", "concat", "max"],
                      help="Ensemble aggregation methods: mean, concat, max, weighted_mean, min")
    parser.add_argument("--similarity_ensemble", action="store_true",
                      help="Use similarity-based ensemble instead of embedding-based ensemble")
    parser.add_argument("--individual_models", action="store_true", default=True,
                      help="Also evaluate individual models (default: True)")
    parser.add_argument("--ensemble_only", action="store_true",
                      help="Evaluate ensemble only, skip individual models")
    
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
    
    # Storage for ensemble evaluation
    all_embeddings = []  # List of (source_emb, target_emb) tuples
    all_similarities = []  # List of similarity matrices
    individual_model_names = []  # Track model names for ensemble
    
    # Evaluate individual models (if requested)
    if not args.ensemble_only:
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
                    mrr, avg_rank, source_emb, target_emb = evaluate_model(
                        model, task_model_name, source_texts, target_texts, task=task
                    )
                    
                    if mrr is not None:
                        results.append({
                            'model': task_model_name,
                            'language_code': args.language_code,
                            'split': args.split,
                            'MRR': mrr,
                            'avg_rank': avg_rank
                        })
                        
                        # Store for ensemble
                        if source_emb is not None and target_emb is not None:
                            all_embeddings.append((source_emb, target_emb))
                            individual_model_names.append(task_model_name)
                            
                            # Calculate and store similarities for similarity-based ensemble
                            from sklearn.metrics.pairwise import cosine_similarity
                            sims = cosine_similarity(source_emb, target_emb)
                            all_similarities.append(sims)
            else:
                # Regular evaluation
                mrr, avg_rank, source_emb, target_emb = evaluate_model(
                    model, model_name, source_texts, target_texts
                )
                
                if mrr is not None:
                    results.append({
                        'model': model_name,
                        'language_code': args.language_code,
                        'split': args.split,
                        'MRR': mrr,
                        'avg_rank': avg_rank
                    })
                    
                    # Store for ensemble
                    if source_emb is not None and target_emb is not None:
                        all_embeddings.append((source_emb, target_emb))
                        individual_model_names.append(model_name)
                        
                        # Calculate and store similarities for similarity-based ensemble
                        from sklearn.metrics.pairwise import cosine_similarity
                        sims = cosine_similarity(source_emb, target_emb)
                        all_similarities.append(sims)
            
            # Clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # Load models only for ensemble evaluation
        for model_name in args.models:
            logger.info(f"Loading model for ensemble: {model_name}")
            model = load_model(model_name)
            
            if model is None:
                logger.warning(f"Skipping model {model_name} due to loading error")
                continue
            
            # Handle special cases for Jina v4 models
            if "jina-embeddings-v4" in model_name:
                # Use both tasks for ensemble
                for task in ['retrieval', 'text-matching']:
                    task_model_name = f"{model_name}_{task}"
                    source_emb = encode_texts(model, source_texts, task=task)
                    target_emb = encode_texts(model, target_texts, task=task)
                    
                    if source_emb is not None and target_emb is not None:
                        all_embeddings.append((source_emb, target_emb))
                        individual_model_names.append(task_model_name)
                        
                        # Calculate similarities
                        from sklearn.metrics.pairwise import cosine_similarity
                        sims = cosine_similarity(source_emb, target_emb)
                        all_similarities.append(sims)
            else:
                # Regular model
                source_emb = encode_texts(model, source_texts)
                target_emb = encode_texts(model, target_texts)
                
                if source_emb is not None and target_emb is not None:
                    all_embeddings.append((source_emb, target_emb))
                    individual_model_names.append(model_name)
                    
                    # Calculate similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    sims = cosine_similarity(source_emb, target_emb)
                    all_similarities.append(sims)
            
            # Clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Evaluate ensemble methods
    if len(all_embeddings) > 1:
        logger.info(f"Evaluating ensemble with {len(all_embeddings)} models")
        
        for method in args.ensemble_methods:
            if args.similarity_ensemble:
                # Similarity-based ensemble
                ensemble_mrr, ensemble_avg_rank = aggregate_similarities(
                    all_similarities, individual_model_names, method
                )
                ensemble_name = f"ensemble_sim_{method}"
            else:
                # Embedding-based ensemble
                ensemble_mrr, ensemble_avg_rank = evaluate_ensemble(
                    all_embeddings, individual_model_names, method
                )
                ensemble_name = f"ensemble_emb_{method}"
            
            if ensemble_mrr is not None:
                results.append({
                    'model': ensemble_name,
                    'language_code': args.language_code,
                    'split': args.split,
                    'MRR': ensemble_mrr,
                    'avg_rank': ensemble_avg_rank
                })
    elif len(all_embeddings) == 1:
        logger.warning("Only one model available for ensemble. Skipping ensemble evaluation.")
    else:
        logger.error("No valid models for ensemble evaluation.")
    
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