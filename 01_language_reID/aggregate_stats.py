import pandas as pd
from glob import glob
import argparse
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def aggregate(stats_dir, output_path):
    all_files = glob(os.path.join(stats_dir, "partial_stats_*.tsv"))
    if not all_files:
        logging.warning("No partial stats files found in {stats_dir}.")
        return

    logging.info(f"Find {len(all_files)} partial stats files to aggregate.")

    df_list = [pd.read_csv(f, sep="\t") for f in all_files]

    full_df = pd.concat(df_list, ignore_index=True)

    agg_df = full_df.groupby('lang_pair').agg({
        'pre_count': 'sum',
        'post_count': 'sum'
    }).reset_index()
    
    agg_df = agg_df.sort_values(by="lang_pair")

    agg_df.to_csv(output_path, sep="\t", index=False)
    logging.info(f"Aggregated stats saved to  {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate partial stats files into a single file.")
    parser.add_argument("--stats_dir", required=True, help="Directory containing partial stats files.")
    parser.add_argument("--output_file", default="filter_stats.tsv", help="Name of the output file.")
    args = parser.parse_args()

    final_path = os.path.join(args.stats_dir, args.output_file)
    
    aggregate(args.stats_dir, final_path)