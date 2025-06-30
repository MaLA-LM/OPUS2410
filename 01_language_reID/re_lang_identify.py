import os
import orjson
import argparse
from glob import glob
from datasets import load_dataset
import fasttext
import sys
import logging
import gzip


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


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
    

def get_lang_preds(source_text, target_text):
    source_pred = lid_model.predict(source_text, 1)
    target_pred = lid_model.predict(target_text, 1)
    return {
        "source_predlang_id": source_pred[0][0].replace("__label__", ""),
        "source_predlang_conf": source_pred[1][0],
        "target_predlang_id": target_pred[0][0].replace("__label__", ""),
        "target_predlang_conf": target_pred[1][0],
    }


def save_jsonl(dataset, path):
    if len(dataset) == 0:
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    total_rows = len(dataset)
    chunk_size = 100000
    buffer_size = 64*1024*1024  # 64MB
    
    with open(path, "wb", buffering=buffer_size) as f:
        for i in range(0, total_rows, chunk_size):
            chunk = dataset[i:i + chunk_size]
            lines = [orjson.dumps(ex, option=orjson.OPT_NON_STR_KEYS) + b"\n" 
                    for ex in chunk]
            f.writelines(lines)
            
            if (i + chunk_size) % 2500000 == 0:
                logging.info(f"Saving Progress: {min(i + chunk_size, total_rows):,}/{total_rows:,}")
    
    logging.info(f"√ Saved to {path}")


def process_file(input_path, output_path, num_proc):
    try:
        ds = load_dataset("json", data_files=input_path, split="train")

        required_keys = {"source_text", "target_text", "source_lang", "target_lang"}
        if not required_keys.issubset(ds.column_names):
            raise ValueError(f"Missing required fields: {required_keys - set(ds.column_names)}")
        
        ds = ds.map(lambda x: get_lang_preds(x["source_text"], x["target_text"]), num_proc=num_proc)
        save_jsonl(ds, output_path)

    except Exception as e:
        logging.error(f"[Error] Failed to process: {input_path}\n{type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .jsonl.gz files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .jsonl files")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--model_path", default="model.bin", help="Path to fastText language ID model")
    parser.add_argument("--filelist", type=str, help="Optional: Path to file containing list of files to process")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Number of Processes: {args.num_proc}")
    logging.info(f"  Model Path: {args.model_path}")
    logging.info(f"  Filelist: {args.filelist}")

    os.makedirs(args.output_dir, exist_ok=True)
    lid_model = fasttext.load_model(args.model_path)

    if args.filelist:
        with open(args.filelist, encoding="utf-8") as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted(glob(f"{args.source_dir}/**/*.jsonl.gz", recursive=True))

    for idx, input_path in enumerate(all_files, 1):
        logging.info(f"[{idx}/{len(all_files)}] Processing file: {os.path.basename(input_path)}")

        rel_path = os.path.relpath(input_path, args.source_dir).replace(".jsonl.gz", "")
        output_path = os.path.join(args.output_dir, rel_path + ".jsonl") 
        
        skip = False
        if os.path.exists(output_path):
            try:
                input_lines = count_lines(input_path)
                output_lines = count_lines(output_path)
                if input_lines == output_lines:
                    logging.info(f"[Skip] {output_path} exists and line count matches ({input_lines})")
                    skip = True
                else:
                    logging.warning(f"[Reprocess] {output_path} exists but line count mismatch (input={input_lines}, output={output_lines})")
            except Exception as e:
                logging.warning(f"[Reprocess] Failed to count lines for {input_path} or {output_path}: {e}")

        if skip:
            continue

        try:
            process_file(input_path, output_path, args.num_proc)
        except Exception as e:
            logging.error(f"[Error] Failed during processing: {input_path}\n{e}")
            continue
