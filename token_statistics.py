import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from typing import Callable, Dict, Optional, List, Any, Tuple
from collections.abc import Iterable

from functools import partial
from copy import copy
import json

from multiprocessing import Pool
from datatools import load, process, identity_fn, ProcessOptions


from pathlib import Path

from contextlib import contextmanager


@dataclass
class DatasetOptions(Serializable):
    """This script requires a strict folder structure where the root folder has
    subfolders for documents, tokens, annotations, domains, each with an equal number of shards.
    Example:
    ```
        base_corpus/
            documents/
                - CC_shard_00000000_processed.jsonl.zst
                - CC_shard_00000001_processed.jsonl.zst
            tokens/
                - CC_shard_00000000_processed.npy
                - CC_shard_00000001_processed.npy
            some_domain_annotation/
                - CC_shard_00000000_processed.npy
                - CC_shard_00000001_processed.npy
    ```
    """
    input_base: Path = field(positional=True, help="Path to the input folder containing labels")

    tokens_dir: Path = field(default="/data/group_data/cx_group/WebOrganizer/Corpus-30B/tokens", help="Relative to the output folder containing tokens")

    document_dir: Path = field(default="documents", help="Relative to the output folder containing tokens")

    token_suffix: str = field(default=".npy", help="Extension of the annotation files")
    document_suffix: str = field(default=".jsonl", help="Extension of the annotation files")
    num_proc: int = field(default=8, help="Number of processes to use", alias="-w")


def load_dataframe(shard_name: Path,
                   options: DatasetOptions):
    token_path =  options.tokens_dir / (shard_name + options.token_suffix)
    df = np.load(token_path)
    return df.sum().item()


def generate_statistics(options: DatasetOptions):
    document_paths = sorted((options.input_base / options.document_dir).glob(f"*{options.document_suffix}"))

    # token_paths = sorted((options.input_base / options.tokens_dir).glob(f"*{options.token_suffix}"))
    shard_names = [
        str(path.name)[:len(str(path.name)) - len(options.document_suffix)-len("_skill")]
        for path in document_paths
    ]
    
    with Pool(processes=options.num_proc) as pool:
        token_sum_list = pool.map(
            partial(load_dataframe, options=options),
            shard_names
        )
        # metadata_df = np.concat(metadata_dfs)
        print(len(token_sum_list))
        print(sum(token_sum_list))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DatasetOptions, dest="options")
    args = parser.parse_args()
    generate_statistics(args.options)
