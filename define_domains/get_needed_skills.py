import json
from tqdm import tqdm
from glob import glob
import os
import numpy as np

input_dir = "/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/documents"
output_dir = "/data/user_data/gonilude/WebOrganizer/Corpus-30B-v2/skills"
os.makedirs(output_dir, exist_ok=True)

files = glob(f"{input_dir}/CC_shard_*.jsonl")
print(f"Found {len(files)} files")

for file in tqdm(files, desc="Processing files"):
    skill_data = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            skill_data.append(data["skill_top_choice_index"])

    skill_data = np.array(skill_data)
    file_name = file.split("/")[-1].split(".")[0]
    shard_id = file_name.split("_")[2]
    output_file = f"{output_dir}/CC_shard_{shard_id}_processed_skill.npy"
    np.save(output_file, skill_data)

print("Done")