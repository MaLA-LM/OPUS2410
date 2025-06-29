import os
import json
import argparse
from glob import glob
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import sys
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def save_stats_table(pre_stats_all, post_stats_all, stats_path):
    all_lang_pairs = set(pre_stats_all) | set(post_stats_all)
    rows = []
    for lp in sorted(all_lang_pairs):
        pre = pre_stats_all.get(lp, 0)
        post = post_stats_all.get(lp, 0)
        rate = post / pre if pre > 0 else 0.0
        rows.append({"lang_pair": lp, "pre_count": pre, "post_count": post, "retention_rate": f"{rate:.2%}"})
    df = pd.DataFrame(rows)
    df.to_csv(stats_path, sep="\t", index=False, mode="w", header=True)
    logging.info(f"[Saved] Language pair filter stats saved to {stats_path}")


def save_jsonl(dataset, path, stats=None):
    if len(dataset) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if stats is not None:
                lang_pair = f"{ex['source_lang']}-{ex['target_lang']}"
                stats[lang_pair] = stats.get(lang_pair, 0) + 1


def count_lines_in_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return count
    except Exception as e:
        logging.error(f"[Error] Failed to count lines in {file_path}: {e}")
        return 0


def filter_file(file_path, output_path, num_proc=8, conf_threshold=0.0, pre_stats=None, post_stats=None):
    try:
        ds = load_dataset("json", data_files=file_path, split="train")

        required_keys = {"source_text", "target_text", "source_lang", "target_lang", 
                        "source_predlang_id", "source_predlang_conf", 
                        "target_predlang_id", "target_predlang_conf"}
        if not required_keys.issubset(ds.column_names):
            raise ValueError(f"Missing required fields: {required_keys - set(ds.column_names)}")
        
        # Count pre-filter stats
        if pre_stats is not None:
            for ex in ds:
                lang_pair = f"{ex['source_lang']}-{ex['target_lang']}"
                pre_stats[lang_pair] = pre_stats.get(lang_pair, 0) + 1

        def lang_match(x):
            return (
                x["source_lang"] == x["source_predlang_id"] and
                x["target_lang"] == x["target_predlang_id"] and
                x["source_predlang_conf"] >= conf_threshold and
                x["target_predlang_conf"] >= conf_threshold
            )

        filtered_ds = ds.filter(lang_match, num_proc=num_proc)
        save_jsonl(filtered_ds, output_path, stats=post_stats)

    except Exception as e:
        logging.error(f"[Error] Failed to filter: {file_path}\n{type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .jsonl files with predictions")
    parser.add_argument("--output_dir", required=True, help="Directory to save filtered .jsonl files")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--conf_threshold", type=float, default=0.0, help="Confidence threshold for filtering")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Number of Processes: {args.num_proc}")
    logging.info(f"  Confidence Threshold: {args.conf_threshold}")

    os.makedirs(args.output_dir, exist_ok=True)
    pre_stats_all = {}
    post_stats_all = {}

    all_files = sorted(glob(f"{args.source_dir}/**/*.jsonl", recursive=True))

    for idx, file_path in enumerate(tqdm(all_files), 1):
        logging.info(f"[{idx}/{len(all_files)}] Processing file: {os.path.basename(file_path)}")

        rel_path = os.path.relpath(file_path, args.source_dir)
        rel_path = rel_path.replace(".jsonl", "")
        output_path = os.path.join(args.output_dir, rel_path + ".jsonl")        

        pre_stats = {}
        post_stats = {}
        
        if os.path.exists(output_path):
            logging.info(f"[Skip] Filter file already processed, loading existing stats: {output_path}")
            lang_pair = rel_path.split('/')[0]
            pre_stats[lang_pair] = count_lines_in_file(file_path)
            post_stats[lang_pair] = count_lines_in_file(output_path)
        else:
            try:
                filter_file(file_path, output_path, args.num_proc, args.conf_threshold,
                           pre_stats=pre_stats, post_stats=post_stats)
            except Exception as e:
                logging.error(f"[Error] Failed during filtering: {file_path}\n{e}")
                continue

        for k, v in pre_stats.items():
            pre_stats_all[k] = pre_stats_all.get(k, 0) + v
        for k, v in post_stats.items():
            post_stats_all[k] = post_stats_all.get(k, 0) + v

    stats_path = os.path.join(args.output_dir, "filter_stats.tsv")
    save_stats_table(pre_stats_all, post_stats_all, stats_path)