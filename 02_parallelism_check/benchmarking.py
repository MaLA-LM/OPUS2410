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

def detect_dataset_type(dataset_name):
    """Detect dataset type based on dataset name"""
    if "flores200" in dataset_name.lower() or "flores" in dataset_name.lower():
        return "flores"
    elif "tatoeba" in dataset_name.lower():
        return "tatoeba"
    else:
        # Try to detect by attempting to load a sample
        logger.warning(f"Unknown dataset type for {dataset_name}, attempting auto-detection...")
        try:
            from datasets import load_dataset
            # Try loading a small sample to detect structure
            sample = load_dataset(dataset_name, split="train[:1]", trust_remote_code=True)
            
            # Check for FLORES structure (sentence_* columns)
            flores_columns = [col for col in sample.column_names if col.startswith('sentence_')]
            if flores_columns:
                logger.info("Detected FLORES-type dataset structure")
                return "flores"
            elif "sourceString" in sample.column_names and "targetString" in sample.column_names:
                logger.info("Detected Tatoeba-type dataset structure")
                return "tatoeba"
            else:
                logger.error(f"Could not detect dataset structure. Available columns: {sample.column_names}")
                return None
        except Exception as e:
            logger.error(f"Failed to auto-detect dataset type: {e}")
            return None

def get_available_languages_flores(dataset_name, split="devtest"):
    """Get all available language codes for FLORES dataset by examining column names"""
    try:
        from datasets import load_dataset
        
        logger.info(f"Getting available languages for FLORES dataset: {dataset_name}")
        
        # Load a small sample with 'all' configuration to get column names
        ds = load_dataset(dataset_name, 'all', split=f"{split}[:1]", trust_remote_code=True)
        
        logger.info(f"Dataset columns: {len(ds.column_names)} total columns")
        
        # Extract language codes from column names that start with 'sentence_'
        language_codes = []
        for column in ds.column_names:
            if column.startswith('sentence_'):
                lang_code = column.replace('sentence_', '')
                language_codes.append(lang_code)
        
        logger.info(f"Found {len(language_codes)} languages in FLORES dataset")
        logger.info(f"First 20 languages: {language_codes[:20]}")
        
        if not language_codes:
            logger.error(f"No language columns found. Available columns: {ds.column_names}")
            
        return language_codes
        
    except Exception as e:
        logger.error(f"Failed to get language configurations for FLORES {dataset_name}: {e}")
        return []

def get_available_languages(dataset_name, dataset_type="auto", split="devtest"):
    """Get all available language codes for the dataset"""
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_name)
    
    # Check if this is a FLORES dataset
    if dataset_type == "flores":
        return get_available_languages_flores(dataset_name, split)
    
    # For other datasets, use the original method
    try:
        from datasets import get_dataset_config_names
        
        configs = get_dataset_config_names(dataset_name)
        logger.info(f"Found {len(configs)} language configurations for {dataset_name}")
        return configs
    except Exception as e:
        logger.error(f"Failed to get language configurations for {dataset_name}: {e}")
        return []

def get_language_pairs_for_flores(languages, source_lang='eng_Latn'):
    """Generate language pairs for FLORES dataset with English as source"""
    pairs = []
    for lang in languages:
        if lang != source_lang:  # Don't pair English with itself
            pairs.append((source_lang, lang))
    return pairs

class MockDataset:
    """Mock dataset class to maintain interface compatibility"""
    def __init__(self, sentences):
        self.data = {"sentence": sentences}
    
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        raise KeyError(f"Column {key} not in the dataset. Available columns: {list(self.data.keys())}")

def load_flores_datasets(dataset_name, source_lang, target_lang, split):
    """Load both source and target language datasets for FLORES"""
    try:
        from datasets import load_dataset
        
        logger.info(f"Loading FLORES dataset: {dataset_name}, split: {split}")
        
        # Load the dataset with 'all' configuration to get all languages
        ds = load_dataset(dataset_name, 'all', split=split, trust_remote_code=True)
        
        logger.info(f"Dataset loaded. Available columns: {ds.column_names}")
        
        # The column names follow the pattern: sentence_{language_code}
        source_column = f"sentence_{source_lang}"
        target_column = f"sentence_{target_lang}"
        
        logger.info(f"Looking for columns: {source_column}, {target_column}")
        
        # Check if the columns exist
        available_columns = ds.column_names
        if source_column not in available_columns:
            # Try to find similar columns
            similar_columns = [col for col in available_columns if col.startswith('sentence_')]
            logger.error(f"Source language column '{source_column}' not found.")
            logger.error(f"Available sentence columns: {similar_columns[:10]}...")  # Show first 10
            raise KeyError(f"Source language column '{source_column}' not found. Available sentence columns: {len(similar_columns)} total")
        
        if target_column not in available_columns:
            # Try to find similar columns
            similar_columns = [col for col in available_columns if col.startswith('sentence_')]
            logger.error(f"Target language column '{target_column}' not found.")
            logger.error(f"Available sentence columns: {similar_columns[:10]}...")  # Show first 10
            raise KeyError(f"Target language column '{target_column}' not found. Available sentence columns: {len(similar_columns)} total")
        
        # Extract the sentences for source and target languages
        source_texts = ds[source_column]
        target_texts = ds[target_column]
        
        logger.info(f"Successfully loaded FLORES dataset: {source_lang} -> {target_lang}, split: {split}")
        logger.info(f"Dataset size: {len(ds)} sentences for each language")
        logger.info(f"Source texts sample: {source_texts[0][:100] if len(source_texts) > 0 else 'Empty'}")
        logger.info(f"Target texts sample: {target_texts[0][:100] if len(target_texts) > 0 else 'Empty'}")
        
        # Return mock datasets with the sentence data
        source_ds = MockDataset(source_texts)
        target_ds = MockDataset(target_texts)
        
        return source_ds, target_ds
        
    except Exception as e:
        logger.error(f"Failed to load FLORES datasets {source_lang} -> {target_lang}: {e}")
        return None, None

def load_tatoeba_dataset(dataset_name, language_code, split):
    """Load Tatoeba dataset with sourceString and targetString columns"""
    try:
        from datasets import load_dataset
        
        ds = load_dataset(dataset_name, language_code, split=split, trust_remote_code=True)
        logger.info(f"Successfully loaded Tatoeba dataset: {dataset_name}, language: {language_code}, split: {split}")
        logger.info(f"Dataset size: {len(ds)}")
        
        # Return source and target texts
        source_texts = ds["sourceString"]
        target_texts = ds["targetString"]
        return source_texts, target_texts
    except Exception as e:
        logger.error(f"Failed to load Tatoeba dataset {dataset_name} with language {language_code}: {e}")
        return None, None

def process_flores_dataset(args, model, output_file):
    """Process FLORES-200 dataset"""
    logger.info("Processing FLORES-200 dataset...")
    
    # Get available languages
    if args.target_languages:
        target_languages = args.target_languages
        logger.info(f"Using specified target languages: {target_languages}")
    else:
        all_languages = get_available_languages(args.dataset_name, "flores", args.split)
        if not all_languages:
            logger.error("Could not retrieve available languages. Exiting.")
            return 0, 0, 0
        # Remove source language from target languages
        target_languages = [lang for lang in all_languages if lang != args.source_lang]
        logger.info(f"Found {len(target_languages)} target languages (excluding source language {args.source_lang})")
    
    # Generate language pairs
    language_pairs = [(args.source_lang, target_lang) for target_lang in target_languages]
    
    # Results storage
    results = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Evaluate across all language pairs
    for source_lang, target_lang in language_pairs:
        logger.info(f"Processing language pair: {source_lang} -> {target_lang} ({processed_count + skipped_count + failed_count + 1}/{len(language_pairs)})")
        
        # Check if already processed
        if args.skip_processed:
            # Handle special cases for Jina v4 models
            if "jina-embeddings-v4" in args.model:
                # Check both task variants
                retrieval_processed = is_language_processed(output_file, f"{args.model}_retrieval", (source_lang, target_lang))
                matching_processed = is_language_processed(output_file, f"{args.model}_text-matching", (source_lang, target_lang))
                if retrieval_processed and matching_processed:
                    logger.info(f"Language pair {source_lang} -> {target_lang} already processed for both tasks. Skipping.")
                    skipped_count += 1
                    continue
            else:
                if is_language_processed(output_file, args.model, (source_lang, target_lang)):
                    logger.info(f"Language pair {source_lang} -> {target_lang} already processed. Skipping.")
                    skipped_count += 1
                    continue
        
        # Load datasets for this language pair
        source_ds, target_ds = load_flores_datasets(args.dataset_name, source_lang, target_lang, args.split)
        if source_ds is None or target_ds is None:
            logger.warning(f"Failed to load datasets for language pair {source_lang} -> {target_lang}. Skipping.")
            failed_count += 1
            continue
        
        # Get source and target texts
        try:
            source_texts = source_ds["sentence"]
            target_texts = target_ds["sentence"]
            logger.info(f"Successfully extracted sentences. Source: {len(source_texts)}, Target: {len(target_texts)}")
        except Exception as e:
            logger.error(f"Failed to extract sentences from mock datasets: {e}")
            failed_count += 1
            continue
        
        # Handle special cases for Jina v4 models
        if "jina-embeddings-v4" in args.model:
            # Evaluate with different tasks
            for task in ['retrieval', 'text-matching']:
                task_model_name = f"{args.model}_{task}"
                
                # Skip if this specific task is already processed
                if args.skip_processed and is_language_processed(output_file, task_model_name, (source_lang, target_lang)):
                    logger.info(f"Language pair {source_lang} -> {target_lang} with task {task} already processed. Skipping.")
                    continue
                
                mrr, avg_rank = evaluate_model(model, task_model_name, source_texts, target_texts, task=task)
                
                if mrr is not None:
                    results.append({
                        'model': task_model_name,
                        'source_lang': source_lang,
                        'target_lang': target_lang,
                        'split': args.split,
                        'MRR': mrr,
                        'avg_rank': avg_rank
                    })
        else:
            # Regular evaluation
            mrr, avg_rank = evaluate_model(model, args.model, source_texts, target_texts)
            
            if mrr is not None:
                results.append({
                    'model': args.model,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'split': args.split,
                    'MRR': mrr,
                    'avg_rank': avg_rank
                })
        
        processed_count += 1
        
        # Save results after each language to avoid losing progress
        if results:
            save_results(output_file, results)
            results = []  # Clear the buffer
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save any remaining results
    if results:
        save_results(output_file, results)
    
    return processed_count, skipped_count, failed_count

def process_tatoeba_dataset(args, model, output_file):
    """Process Tatoeba dataset"""
    logger.info("Processing Tatoeba dataset...")
    
    # Get available languages
    if args.target_languages:
        languages = args.target_languages
        logger.info(f"Using specified languages: {languages}")
    else:
        languages = get_available_languages(args.dataset_name, "tatoeba")
        if not languages:
            logger.error("Could not retrieve available languages. Exiting.")
            return 0, 0, 0
    
    # Results storage
    results = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Evaluate across all languages
    for language_code in languages:
        logger.info(f"Processing language: {language_code} ({processed_count + skipped_count + failed_count + 1}/{len(languages)})")
        
        # Check if already processed
        if args.skip_processed:
            # Handle special cases for Jina v4 models
            if "jina-embeddings-v4" in args.model:
                # Check both task variants
                retrieval_processed = is_language_processed(output_file, f"{args.model}_retrieval", language_code)
                matching_processed = is_language_processed(output_file, f"{args.model}_text-matching", language_code)
                if retrieval_processed and matching_processed:
                    logger.info(f"Language {language_code} already processed for both tasks. Skipping.")
                    skipped_count += 1
                    continue
            else:
                if is_language_processed(output_file, args.model, language_code):
                    logger.info(f"Language {language_code} already processed. Skipping.")
                    skipped_count += 1
                    continue
        
        # Load dataset for this language
        source_texts, target_texts = load_tatoeba_dataset(args.dataset_name, language_code, args.split)
        if source_texts is None or target_texts is None:
            logger.warning(f"Failed to load dataset for language {language_code}. Skipping.")
            failed_count += 1
            continue
        
        # Handle special cases for Jina v4 models
        if "jina-embeddings-v4" in args.model:
            # Evaluate with different tasks
            for task in ['retrieval', 'text-matching']:
                task_model_name = f"{args.model}_{task}"
                
                # Skip if this specific task is already processed
                if args.skip_processed and is_language_processed(output_file, task_model_name, language_code):
                    logger.info(f"Language {language_code} with task {task} already processed. Skipping.")
                    continue
                
                mrr, avg_rank = evaluate_model(model, task_model_name, source_texts, target_texts, task=task)
                
                if mrr is not None:
                    results.append({
                        'model': task_model_name,
                        'language_code': language_code,
                        'split': args.split,
                        'MRR': mrr,
                        'avg_rank': avg_rank
                    })
        else:
            # Regular evaluation
            mrr, avg_rank = evaluate_model(model, args.model, source_texts, target_texts)
            
            if mrr is not None:
                results.append({
                    'model': args.model,
                    'language_code': language_code,
                    'split': args.split,
                    'MRR': mrr,
                    'avg_rank': avg_rank
                })
        
        processed_count += 1
        
        # Save results after each language to avoid losing progress
        if results:
            save_results(output_file, results)
            results = []  # Clear the buffer
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save any remaining results
    if results:
        save_results(output_file, results)
    
    return processed_count, skipped_count, failed_count

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

def is_language_processed(output_file, model_name, identifier):
    """Check if a language/language pair has already been processed for the given model
    
    Args:
        output_file: Path to the results file
        model_name: Name of the model
        identifier: Either language_code (for Tatoeba) or tuple (source_lang, target_lang) for FLORES
    """
    if not output_file.exists():
        return False
    
    try:
        df = pd.read_csv(output_file)
        
        if isinstance(identifier, tuple):
            # FLORES dataset - check source_lang and target_lang
            source_lang, target_lang = identifier
            mask = (df['model'] == model_name) & (df['source_lang'] == source_lang) & (df['target_lang'] == target_lang)
        else:
            # Tatoeba dataset - check language_code
            language_code = identifier
            mask = (df['model'] == model_name) & (df['language_code'] == language_code)
        
        return mask.any()
    except Exception as e:
        logger.warning(f"Could not read existing results file: {e}")
        return False

def sanitize_model_name(model_name):
    """Sanitize model name for use in filename"""
    # Replace problematic characters with underscores
    sanitized = model_name.replace('/', '_').replace(':', '_').replace('-', '_')
    # Remove any other problematic characters
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    # Remove consecutive underscores and strip leading/trailing underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_')

def save_results(output_file, new_results):
    """Save results to CSV, appending to existing file if it exists"""
    df_new = pd.DataFrame(new_results)
    
    if output_file.exists():
        try:
            df_existing = pd.read_csv(output_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            logger.warning(f"Could not read existing file, creating new one: {e}")
            df_combined = df_new
    else:
        df_combined = df_new
    
    df_combined.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence transformer model on translation retrieval across all languages")
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="Dataset name (e.g., 'Muennighoff/flores200' or 'Helsinki-NLP/tatoeba_mt')")
    parser.add_argument("--model", type=str, required=True,
                      help="Pretrained sentence transformer model to evaluate")
    parser.add_argument("--split", type=str, default=None,
                      help="Dataset split (default: auto-detect based on dataset)")
    parser.add_argument("--source_lang", type=str, default="eng_Latn",
                      help="Source language code for FLORES dataset (default: 'eng_Latn' for English)")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Output directory for results (default: './results')")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for encoding (default: 32)")
    parser.add_argument("--skip_processed", action="store_true",
                      help="Skip languages/language pairs that have already been processed")
    parser.add_argument("--target_languages", type=str, nargs="*",
                      help="Specific target languages to evaluate (if not provided, evaluates all)")
    
    args = parser.parse_args()
    
    # Detect dataset type
    dataset_type = detect_dataset_type(args.dataset_name)
    if dataset_type is None:
        logger.error("Could not detect dataset type. Exiting.")
        sys.exit(1)
    
    # Set default split based on dataset type if not specified
    if args.split is None:
        if dataset_type == "flores":
            args.split = "devtest"
        elif dataset_type == "tatoeba":
            args.split = "test"
        logger.info(f"Using default split for {dataset_type} dataset: {args.split}")
    
    # Setup environment
    setup_environment()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename with sanitized names
    dataset_basename = args.dataset_name.split('/')[-1] if '/' in args.dataset_name else args.dataset_name
    sanitized_model_name = sanitize_model_name(args.model)
    output_file = output_dir / f"{dataset_basename}_{sanitized_model_name}.csv"
    
    # Load model once
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model)
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Process dataset based on type
    if dataset_type == "flores":
        processed_count, skipped_count, failed_count = process_flores_dataset(args, model, output_file)
        total_items = f"language pairs"
    elif dataset_type == "tatoeba":
        processed_count, skipped_count, failed_count = process_tatoeba_dataset(args, model, output_file)
        total_items = f"languages"
    
    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_name} ({dataset_type.upper()})")
    if dataset_type == "flores":
        print(f"Source Language: {args.source_lang}")
    print(f"Total {total_items}: {processed_count + skipped_count + failed_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {output_file}")
    
    # Display final results
    if output_file.exists():
        try:
            df = pd.read_csv(output_file)
            # Filter for current model results
            if "jina-embeddings-v4" in args.model:
                model_results = df[df['model'].str.startswith(args.model)]
            else:
                model_results = df[df['model'] == args.model]
            
            if not model_results.empty:
                print("\nFINAL RESULTS:")
                print(model_results.to_string(index=False))
        except Exception as e:
            logger.error(f"Could not display final results: {e}")
    
    print("="*80)

if __name__ == "__main__":
    main()