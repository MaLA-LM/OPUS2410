import os
import gzip
import argparse
from glob import glob
from datasets import load_dataset
import pandas as pd
import sys
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def save_stats_table(pre_stats_all, post_stats_all, stats_path, append=False):
    all_lang_pairs = set(pre_stats_all) | set(post_stats_all)
    rows = []
    for lp in sorted(all_lang_pairs):
        pre = pre_stats_all.get(lp, 0)
        post = post_stats_all.get(lp, 0)
        rate = post / pre if pre > 0 else 0.0
        rows.append({"lang_pair": lp, "pre_count": pre, "post_count": post, "retention_rate": f"{rate:.2%}"})
    df = pd.DataFrame(rows)
    df.to_csv(stats_path, sep="\t", index=False, mode="a" if append else "w", header=not append)
    logging.info(f"[Saved] Language pair filter stats saved to {stats_path}")


def save_jsonl(dataset, path):
    if len(dataset) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_json(path, lines=True)
    logging.info(f"√ Saved to {path}")


def count_lines(file_path):
    def _count_newlines(data):
        return data.count(b'\n')  # Count newline bytes

    open_fn = gzip.open if file_path.endswith('.gz') else open
    total_lines = 0
    with open_fn(file_path, 'rb') as f:  # Read in binary mode
        while True:
            chunk = f.read(64*1024*1024)  # Read 64MB chunks
            if not chunk:
                break
            total_lines += _count_newlines(chunk)
    return total_lines


def filter_file(input_path, output_path, num_proc=8, conf_threshold=0.0, pre_stats=None, post_stats=None, lang_pair=None):
    try:
        ds = load_dataset("json", data_files=input_path, split="train")

        required_keys = {"source_text", "target_text", "source_lang", "target_lang", 
                        "source_predlang_id", "source_predlang_conf", 
                        "target_predlang_id", "target_predlang_conf"}
        if not required_keys.issubset(ds.column_names):
            raise ValueError(f"Missing required fields: {required_keys - set(ds.column_names)}")
        
        # Count pre-filter stats
        pre_stats[lang_pair] = count_lines(input_path)

        def lang_match(x):
            return (
                x["source_lang"] == x["source_predlang_id"] and
                x["target_lang"] == x["target_predlang_id"] and
                x["source_predlang_conf"] >= conf_threshold and
                x["target_predlang_conf"] >= conf_threshold
            )

        filtered_ds = ds.filter(lang_match, num_proc=num_proc)
        save_jsonl(filtered_ds, output_path)

        # Count post-filter stats
        if os.path.exists(output_path):
            post_stats[lang_pair] = count_lines(output_path)
        else:
            post_stats[lang_pair] = 0

    except Exception as e:
        logging.error(f"[Error] Failed to filter: {input_path}\n{type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .jsonl files with predictions")
    parser.add_argument("--output_dir", required=True, help="Directory to save filtered .jsonl files")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--conf_threshold", type=float, default=0.0, help="Confidence threshold for filtering")
    parser.add_argument("--filelist", type=str, help="Optional: Path to file containing list of files to process")
    parser.add_argument("--job_id", type=str, help="Optional: Job ID")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Number of Processes: {args.num_proc}")
    logging.info(f"  Confidence Threshold: {args.conf_threshold}")
    logging.info(f"  Filelist: {args.filelist}")

    os.makedirs(args.output_dir, exist_ok=True)
    pre_stats_all = {}
    post_stats_all = {}

    if args.filelist:
        with open(args.filelist, encoding="utf-8") as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted(glob(f"{args.source_dir}/**/*.jsonl", recursive=True))

    for idx, input_path in enumerate(all_files, 1):
        logging.info(f"[{idx}/{len(all_files)}] Processing file: {os.path.basename(input_path)}")

        rel_path = os.path.relpath(input_path, args.source_dir)
        rel_path = rel_path.replace(".jsonl", "")
        output_path = os.path.join(args.output_dir, rel_path + ".jsonl")
        lang_pair = rel_path.split('/')[0]       

        pre_stats = {}
        post_stats = {}
        
        if os.path.exists(output_path):
            logging.info(f"[Skip] Filter file already processed, loading existing stats: {output_path}")
            pre_stats[lang_pair] = count_lines(input_path)
            post_stats[lang_pair] = count_lines(output_path)
        else:
            try:
                filter_file(input_path, output_path, args.num_proc, args.conf_threshold,
                           pre_stats=pre_stats, post_stats=post_stats, lang_pair=lang_pair)
            except Exception as e:
                logging.error(f"[Error] Failed during filtering: {input_path}\n{e}")
                continue

        for k, v in pre_stats.items():
            pre_stats_all[k] = pre_stats_all.get(k, 0) + v
        for k, v in post_stats.items():
            post_stats_all[k] = post_stats_all.get(k, 0) + v

    if args.job_id:
        stats_filename = f"partial_stats_{args.job_id}.tsv"
    else:
        stats_filename = "filter_stats.tsv"

    # Create a directory for partial stats
    partial_stats_dir = os.path.join(args.output_dir, "partial_stats")
    os.makedirs(partial_stats_dir, exist_ok=True)
    stats_path = os.path.join(partial_stats_dir, stats_filename)
    # append_mode = os.path.exists(stats_path)
    save_stats_table(pre_stats_all, post_stats_all, stats_path, append=False)