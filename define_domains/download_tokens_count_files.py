from huggingface_hub import hf_hub_download
from glob import glob
import os
from time import sleep

output_dir = "/data/group_data/cx_group/WebOrganizer/Corpus-30B"
os.makedirs(output_dir, exist_ok=True)
from multiprocessing import Pool


def read_file(dir):
    files = glob("/data/group_data/cx_group/WebOrganizer/Corpus-30B/documents/CC_shard_*.jsonl")
    shard_ids = []

    for file in files:
        file_name = file.split("/")[-1].split(".")[0]
        shard_id = file_name.split("_")[2]
        shard_ids.append(shard_id)
    return shard_ids



def download_file(shard_id, ):
    hf_hub_download('WebOrganizer/Corpus-200B', filename=f'tokens/CC_shard_{shard_id}_processed.npy', local_dir=output_dir, repo_type='dataset')


def process_fn(shard_id):
    download_file(shard_id)


if __name__ == "__main__":
    shard_ids = read_file(output_dir)

    # with Pool(processes=24) as pool:
    #     pool.map(process_fn, shard_ids)

    for shard_id in shard_ids:
        output_file = f"{output_dir}/tokens/CC_shard_{shard_id}_processed.npy"
        if os.path.exists(output_file):
            continue

        download_file(shard_id)
        sleep(2)




