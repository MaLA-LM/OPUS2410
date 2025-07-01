import os
from glob import glob

source_dir = "/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410"
output_dir = "./filelists"
num_splits = 128

os.makedirs(output_dir, exist_ok=True)

all_files = sorted(glob(f"{source_dir}/**/*.jsonl.gz", recursive=True))
file_sizes = [(path, os.path.getsize(path)) for path in all_files]

file_sizes.sort(key=lambda x: x[1], reverse=True)

# Allocate the large files to the shard with the lightest load first
partitions = [[] for _ in range(num_splits)]
partition_loads = [0] * num_splits

for path, size in file_sizes:
    idx = partition_loads.index(min(partition_loads))
    partitions[idx].append(path)
    partition_loads[idx] += size

for i, chunk in enumerate(partitions):
    with open(os.path.join(output_dir, f"filelist_{i}.txt"), "w", encoding="utf-8") as f:
        for path in chunk:
            f.write(path + "\n")
