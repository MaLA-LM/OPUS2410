import os
import json
import argparse
from glob import glob
from datasets import load_dataset
import fasttext
from tqdm import tqdm
import pandas as pd


def save_stats_table(pre_stats_all, post_stats_all, out_path):
    all_lang_pairs = set(pre_stats_all) | set(post_stats_all)
    rows = []
    for lp in sorted(all_lang_pairs):
        pre = pre_stats_all.get(lp, 0)
        post = post_stats_all.get(lp, 0)
        rate = post / pre if pre > 0 else 0.0
        rows.append({"lang_pair": lp, "pre_count": pre, "post_count": post, "retention_rate": f"{rate:.2%}"})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[Saved] Language pair stats saved to {out_path}")


def get_lang_preds(source_text, target_text):
    source_pred = lid_model.predict(source_text, 1)
    target_pred = lid_model.predict(target_text, 1)
    return {
        "source_predlang_id": source_pred[0][0].replace("__label__", ""),
        "source_predlang_conf": source_pred[1][0],
        "target_predlang_id": target_pred[0][0].replace("__label__", ""),
        "target_predlang_conf": target_pred[1][0],
    }


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


def process_file(file_path, source_dir, output_dir, num_proc=8, conf_threshold=0.0, pre_stats=None, post_stats=None):
    try:
        rel_path = os.path.relpath(file_path, source_dir).replace(".jsonl.gz", "")
        pre_path = os.path.join(output_dir, "pre_filter", rel_path + ".pre_filter.jsonl")
        post_path = os.path.join(output_dir, "filtered", rel_path + ".filtered.jsonl")

        ds = load_dataset("json", data_files=file_path, split="train")

        required_keys = {"source_text", "target_text", "source_lang", "target_lang"}
        if not required_keys.issubset(ds.column_names):
            raise ValueError(f"Missing required fields: {required_keys - set(ds.column_names)}")
        
        ds = ds.map(lambda x: get_lang_preds(x["source_text"], x["target_text"]), num_proc=num_proc)
        save_jsonl(ds, pre_path, stats=pre_stats)

        def lang_match(x):
            return (
                x["source_lang"] == x["source_predlang_id"] and
                x["target_lang"] == x["target_predlang_id"] and
                x["source_predlang_conf"] >= conf_threshold and
                x["target_predlang_conf"] >= conf_threshold
            )

        filtered_ds = ds.filter(lang_match, num_proc=num_proc)
        save_jsonl(filtered_ds, post_path, stats=post_stats)

    except Exception as e:
        print(f"[Error] Failed to process: {file_path}\n{type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .jsonl.gz files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .jsonl files")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--model_path", default="model.bin", help="Path to fastText language ID model")
    parser.add_argument("--conf_threshold", type=float, default=0.0, help="Confidence threshold for filtering")
    args = parser.parse_args()

    print("Arguments:")
    print(f"  Source Directory: {args.source_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Number of Processes: {args.num_proc}")
    print(f"  Model Path: {args.model_path}")
    print(f"  Confidence Threshold: {args.conf_threshold}")

    os.makedirs(args.output_dir, exist_ok=True)
    lid_model = fasttext.load_model(args.model_path)
    pre_stats_all = {}
    post_stats_all = {}

    all_files = sorted(glob(f"{args.source_dir}/**/*.jsonl.gz", recursive=True))
    for file_path in tqdm(all_files):
        rel_path = os.path.relpath(file_path, args.source_dir).replace(".jsonl.gz", ".jsonl")
        post_filter_path = os.path.join(args.output_dir, "filtered", rel_path + ".filtered.jsonl")

        if os.path.exists(post_filter_path):
            print(f"[Skip] Already exists: {post_filter_path}")
            continue

        try:
            pre_stats = {}
            post_stats = {}
            process_file(file_path, args.source_dir, args.output_dir, args.num_proc, args.conf_threshold,
                         pre_stats=pre_stats, post_stats=post_stats)
            
            for k, v in pre_stats.items():
                pre_stats_all[k] = pre_stats_all.get(k, 0) + v
            for k, v in post_stats.items():
                post_stats_all[k] = post_stats_all.get(k, 0) + v

        except Exception as e:
            print(f"[Error] Failed during processing: {file_path}\n{e}")

    stats_path = os.path.join(args.output_dir, "langpair_stats.tsv")
    save_stats_table(pre_stats_all, post_stats_all, stats_path)
