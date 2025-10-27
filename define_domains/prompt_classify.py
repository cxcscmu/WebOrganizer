"""
Usage:
OUTLINES_CACHE_DIR=/tmp/outlines python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000
python prompt_classify  <input datasets>  <output dataset>  --config_path <config>
"""

import sglang as sgl
import json
from tqdm import tqdm
import numpy as np
import torch

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import time

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from retry import retry
from urllib.error import URLError
import os


def get_excel_col(n: int) -> str:
    """
    Converts a positive integer to an Excel-style column name.
    1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, 28 -> AB, ...
    """
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name

@dataclass
class PromptConfig(Serializable):
    system_template: Optional[str] = None
    template: Optional[str] = None
    choices: Optional[List[str]] = None
    demonstrations: List[Dict[str, str]] = field(cmd=False, default=None)
    labels: List[str] = field(default_factory=lambda: [get_excel_col(i + 1) for i in range(48)])

    response_prefix: Optional[str] = ""
    truncation: int = 50_000  # Truncate the input text to this character length

    randomize_choices: bool = True
    randomize_demonstrations: bool = True
    randomize_seed: int = 42


def get_permutation(index: int, prompt_config: PromptConfig) -> np.ndarray:
    if not prompt_config.randomize_choices:
        return np.arange(len(prompt_config.choices))
    else:
        np.random.seed(index + prompt_config.randomize_seed)
        permutation = np.random.permutation(len(prompt_config.choices))
        return permutation


def get_demonstration_permutation(index: int, prompt_config: PromptConfig) -> np.ndarray:
    if not prompt_config.randomize_demonstrations:
        return np.arange(len(prompt_config.demonstrations))
    else:
        np.random.seed(index + prompt_config.randomize_seed + 1)
        permutation = np.random.permutation(len(prompt_config.demonstrations))
        return permutation


@sgl.function
def classify(s, item: Dict[str, Any], index: int, prompt_config: PromptConfig):
    permutation = get_permutation(index, prompt_config)
    labels = prompt_config.labels[:len(prompt_config.choices)]
    choices = "\n".join(f"{labels[j]}: {prompt_config.choices[i]}" for j, i in enumerate(permutation))

    kwargs = item.copy()
    if len(kwargs["text"]) > prompt_config.truncation:
        kwargs["text"] = kwargs["text"][:prompt_config.truncation] + "... (truncated)"
    kwargs["choices"] = choices

    if prompt_config.system_template is not None:
        s += sgl.system(prompt_config.system_template.format(**kwargs))
    prompt = prompt_config.template.format(**kwargs)

    if prompt_config.demonstrations is not None:
        demonstration_permutation = get_demonstration_permutation(index, prompt_config)
        for j in demonstration_permutation:
            demonstration = prompt_config.demonstrations[j]

            kwargs = demonstration.copy()
            if len(kwargs["text"]) > prompt_config.truncation:
                kwargs["text"] = kwargs["text"][:prompt_config.truncation] + "... (truncated)"
            kwargs["choices"] = choices

            label_index = next(i for i, v in enumerate(prompt_config.choices) if v.startswith(demonstration["choice"]))
            permuted_label_index = np.where(permutation == label_index)[0][0]
            label = labels[permuted_label_index]

            s += sgl.user(prompt_config.template.format(**kwargs))
            if "explanation" in demonstration:
                s += sgl.assistant(prompt_config.response_prefix + label + ": " + demonstration["explanation"])
            else:
                s += sgl.assistant(prompt_config.response_prefix + label)


    s += sgl.user(prompt)
    s += sgl.assistant(prompt_config.response_prefix + sgl.gen("choice", choices=labels))


@retry(URLError, tries=360, backoff=1, delay=5)
def set_default_backend(port=30000):
    sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}"))


def predict_fn(dataset,
               indices,
               process_id,
               prompt_config,
               num_threads=1,
               batch_size=1000,
               port=30000,
               output_path='output.jsonl'):
    set_default_backend(port)

    # Check if output file exists and count existing records
    start_index = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                # Count lines in the file to determine how many records exist
                existing_count = sum(1 for line in f if line.strip())
            print(f"Found existing output file with {existing_count} records. Resuming from index {existing_count}")
            start_index = existing_count
            
            # If we've already processed all data, return early
            if start_index >= len(dataset):
                print(f"All {len(dataset)} records already processed. Nothing to do.")
                return        
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            print("Starting from the beginning...")
            start_index = 0
    else:
        print(f"No existing output file found. Starting from the beginning.")


    start_time = time.time()
    for batch_start in range(start_index, len(dataset), batch_size):
        batch_range = list(range(batch_start, min(batch_start + batch_size, len(dataset))))
        print(f"Processing batch {batch_range[0]} - {batch_range[-1]}")

        states = classify.run_batch([
            {"item": dataset[i], "index": indices[i], "prompt_config": prompt_config}
            for i in batch_range
        ], num_threads=num_threads, progress_bar=True)

        # Check for corruption of inference server
        for state in states:
            meta_info = state.get_meta_info("choice")

            assert meta_info is not None and meta_info["normalized_prompt_logprobs"] is not None
            assert all(
                len(answer_tokens) > 1 for answer_tokens in meta_info["input_token_logprobs"]
            ), f"All answers should have at least 2 tokens in {meta_info['input_token_logprobs']}"


        prediction_results = []
        for i, state in zip(batch_range, states):
            demonstration_permutation = get_demonstration_permutation(indices[i], prompt_config)
            permutation = get_permutation(indices[i], prompt_config)
            meta_info = state.get_meta_info("choice")

            # We re-compute answer logprobs, as the first token is the preceding token
            # that is the same for all answers
            permuted_choice_loss = np.array([
                sum(logprob for logprob, token_id, _ in answer_tokens[1:]) / (len(answer_tokens) - 1)
                for answer_tokens in meta_info["input_token_logprobs"]
            ])

            choice_loss = np.zeros_like(permuted_choice_loss)
            choice_loss[permutation] = permuted_choice_loss

            scores = choice_loss - np.max(choice_loss)
            scores = scores - np.log(np.exp(scores).sum())
            probs = np.exp(scores)

            prediction = np.argmax(probs)

            prediction_results.append( {
                **dataset[i],
                "skill_choice_loss": choice_loss.tolist(),
                "skill_choice_probs": probs.tolist(),
                "skill_top_choice": prompt_config.choices[prediction],
                "skill_top_choice_index": int(prediction),
                "skill_top_choice_prob": float(probs[prediction]),
                "skill_label_permutation": permutation.tolist(),
                "skill_fewshot_permutation": demonstration_permutation.tolist(),
            })
        
        with open(output_path, 'a') as f:
            for output in prediction_results:
                # print(output)
                json.dump(output, f)
                f.write('\n')
    
    print(f"Time taken: {time.time() - start_time:.2f}s")

    job_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if job_id:
        with open(f'/tmp/job_exit_{job_id}.flag', 'w') as f:
            f.write('exit')


def load_dataset(dataset_path, shard_id, num_shards, no_sharding=False):
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            datapoint.pop("predicted_label_id", None)
            data.append(datapoint)
    
    if no_sharding:
        return data

    shard_size = len(data) // num_shards
    start_index = shard_id * shard_size
    end_index = (shard_id + 1) * shard_size if (shard_id + 1) < num_shards else len(data)
    data_shard = data[start_index:end_index]

    print(f"Loaded shard {shard_id} with {len(data_shard)} samples")
    print(f"Data indice from {start_index} to {end_index}")
    return data_shard

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of threads to use")
    parser.add_argument("--port", type=int, default=30000, help="Number of threads to use")
    parser.add_argument("--randomize_seed", default=None, type=int, help="Seed for randomization")
    parser.add_argument("--job_id", default=None, type=int, help="Seed for randomization")
    parser.add_argument("--num_jobs", default=None, type=int, help="Seed for randomization")
    parser.add_argument("--no_sharding", default=False, action="store_true", help="Seed for randomization")


    # parser.add_arguments(LoadOptions, dest="load_options")
    # parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    prompt_config = PromptConfig.load_yaml(args.config_path)

    if args.randomize_seed is not None:
        prompt_config.randomize_seed = args.randomize_seed
    
    args.prompt_config = prompt_config

    print("Arguments:", args)
    # dataset = load(*args.inputs, options=args.load_options)
    dataset = load_dataset(args.inputs[0], args.job_id, args.num_jobs, no_sharding=args.no_sharding)

    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    predict_fn(dataset, list(range(N)), 0, prompt_config, args.num_threads, args.batch_size, args.port, args.output)
    
    # process(
    #     dataset,
    #     partial(
    #         predict_fn,
    #         prompt_config=prompt_config,
    #         num_threads=args.num_threads,
    #         batch_size=args.batch_size,
    #         port=args.port
    #     ),
    #     args.output, args.process_options
    # )
