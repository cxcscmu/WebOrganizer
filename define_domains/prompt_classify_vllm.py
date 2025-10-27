
import json
# from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import numpy as np
import torch

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

from datatools.process import process, ProcessOptions
from datatools.load import load, LoadOptions
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import Serializable
from dotenv import load_dotenv

load_dotenv() 




# Load the LLM model and tokenizer
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"  # Or another suitable model

def load_model(model_name):
    
    # Load the default sampling parameters from the model.
    # llm = LLM(model=model_name, download_dir='/data/group_data/cx_group/', tensor_parallel_size=2)
    # sampling_params = llm.get_default_sampling_params()
    # sampling_params.max_tokens = 1
    # sampling_params.temperature =0.0
    # sampling_params.logprobs = 10
    # sampling_params.prompt_logprobs = 15
    # sampling_params.n = 1
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = ''
    sampling_params = ''
    tokenizer = ''
    return llm, sampling_params, tokenizer


def load_dataset(dataset_path, shard_id, num_shards):
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            data.append(datapoint)

    shard_size = len(data) // num_shards
    start_index = shard_id * shard_size
    end_index = (shard_id + 1) * shard_size if (shard_id + 1) < num_shards else len(data)
    data_shard = data[start_index:end_index]

    print(f"Loaded shard {shard_id} with {len(data_shard)} samples")
    print(f"Data indice from {start_index} to {end_index}")
    return data_shard


def get_choice_token_ids(tokenizer, labels):
    """Get token IDs for choice labels (A, B, C, D, etc.)"""
    choice_token_ids = {}
    for label in labels:
        
        tokens = tokenizer.encode(label, add_special_tokens=False)
        choice_token_ids[label] = tokens[0]
    
    return choice_token_ids

def extract_choice_logprobs(outputs, choice_token_ids, tokenizer):
    """Extract logprobs for each choice from vLLM output"""
    results = []
    
    for output in outputs:
        choice_logprobs = {}
        generated_token_id = None
        generated_choice = None
        
        if output.outputs[0].logprobs:
            token_logprobs = output.outputs[0].logprobs[0]
            generated_token_id = token_logprobs.token_id
            
            # Map token IDs back to choice labels
            id_to_choice = {v: k for k, v in choice_token_ids.items()}
            
            # Get logprobs for all choice tokens
            for token_id, logprob in token_logprobs.top_logprobs.items():
                if token_id in id_to_choice:
                    choice_label = id_to_choice[token_id]
                    choice_logprobs[choice_label] = logprob.logprob
            
            # Determine which choice was generated
            if generated_token_id in id_to_choice:
                generated_choice = id_to_choice[generated_token_id]
            
            # Fill in missing choices with very low probability
            for label in choice_token_ids.keys():
                if label not in choice_logprobs:
                    choice_logprobs[label] = -100.0
        
        results.append({
            'generated_choice': generated_choice,
            'choice_logprobs': choice_logprobs,
            'generated_token_id': generated_token_id
        })
    
    return results


def template_with_data(datapoint, tokenizer, prompt_config, index):
    
    permutation = get_permutation(index, prompt_config)
    labels = prompt_config.labels[:len(prompt_config.choices)]
    choices = "\n".join(f"{labels[j]}: {prompt_config.choices[i]}" for j, i in enumerate(permutation))

    kwargs = datapoint.copy()
    if len(kwargs["text"]) > prompt_config.truncation:
        kwargs["text"] = kwargs["text"][:prompt_config.truncation] + "... (truncated)"
    kwargs["choices"] = choices

    if prompt_config.system_template is not None:
        system_prompt = prompt_config.system_template.format(**kwargs)
    
    prompt = prompt_config.template.format(**kwargs)

    fewshot = []
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

            few_shot_user_prompt = prompt_config.template.format(**kwargs) + "\n"
            if "explanation" in demonstration:
                fewshot_ass_prompt = (prompt_config.response_prefix + label + ": " + demonstration["explanation"] + "\n")
            else:
                fewshot_ass_prompt = (prompt_config.response_prefix + label + "\n")
            
            fewshot.extend([{"role": "user", "content": few_shot_user_prompt}, {"role": "assistant", "content": fewshot_ass_prompt}])
            

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *fewshot,
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # chat_template = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    return messages, permutation


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

def convert_letter_to_number(label):
    assert len(label) == 1 or len(label) == 2
    
    index = 0 if len(label) == 1 else 26
    for char in label:
        index += ord(char.upper()) - ord('A')
    return index


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


def predict_fn(dataset,
               tokenizer,
               llm,
               sampling_params,
               prompt_config):

    start_time = time.time()

    prompts = []
    choice_permutations = []
    for index, data in enumerate(dataset):
        prompt, permutation = template_with_data(data, tokenizer, prompt_config, index)
        prompts.append(prompt)
        choice_permutations.append(permutation)

    # outputs = llm.generate(prompts, sampling_params)
    return prompts, choice_permutations




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("inputs", type=Path, help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--randomize_seed", default=None, type=int, help="Seed for randomization")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=24)

    args = parser.parse_args()
    prompt_config = PromptConfig.load_yaml(args.config_path)

    if args.randomize_seed is not None:
        prompt_config.randomize_seed = args.randomize_seed

    args.prompt_config = prompt_config

    llm, sampling_params, tokenizer = load_model(args.model_name)
    # print("Arguments:", args)
    print("Loading dataset...")
    dataset = load_dataset(args.inputs, args.shard_id, args.num_shards)
    indices = list(range(len(dataset)))

    print("Generating prompts...")
    start_time = time.time()
    prompts, cp = predict_fn(dataset, tokenizer, llm, sampling_params, prompt_config)
    print(f"Prompts generated in {time.time() - start_time} seconds")
    # print(prompts[0])
    # print(cp[0])
    # predict_label = "C"
    # top_choice_index = cp[0][convert_letter_to_number(predict_label)]
    # print(top_choice_index)
    # top_choice = prompt_config.choices[top_choice_index]
    # print(top_choice)

    # import anthropic
    import together
    import random
    import time

    # client = anthropic.Anthropic()
    client = together.Together()

    # comparison_file = 'datasets/dclm-sample980K-skills-8B-seed43-job0.jsonl'
    # print(f"Loading comparison data from {comparison_file}...")
    # with open(comparison_file, 'r') as f:
    #     small_model_predictions_data = [json.loads(line) for line in f]
    # print("Comparison data loaded.")

    random.seed(42)
    # num_samples = min(len(prompts), len(small_model_predictions_data))
    random_indices = random.sample(range(len(dataset)), 10_000)
    sanity_check = {}
    skills_seen = set()
    counter = 0
    skills_needed = set(["Parts of Speech Recognition and Tagging", "Contradiction and Entailment Recognition"])
    random_indices_part_2 = random.sample(range(len(dataset)), 20_000)
    random_indices_part_1 = set(random_indices)
    with open("sanity_check_sample-10K_model-llama70B.jsonl", "a") as f:
        
        for i, prompt_index in enumerate(tqdm(random_indices_part_2, desc="Sanity checking with Claude")):
            if prompt_index in random_indices_part_1:
                continue
            # system_prompt = prompts[prompt_index][0]
            # messages = prompts[prompt_index][1:]

            # response = client.messages.create(
            #     model="claude-opus-4-20250514",
            #     max_tokens=10,
            #     temperature=0,
            #     system=system_prompt['content'],
            #     messages=messages
            # )

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                max_tokens=5,
                temperature=0,
                messages=prompts[prompt_index]
            )

            result = response.choices[0].message.content

            # result = response.content[0].text
            predict_label = result.split(":")[0].strip()
            
            top_choice_index = cp[prompt_index][convert_letter_to_number(predict_label)]
            top_choice = prompt_config.choices[top_choice_index].split("\n")[0].strip()
            skills_seen.add(top_choice)

            if top_choice in skills_needed:
                print(top_choice)
            

            # data = small_model_predictions_data[prompt_index]
            # small_model_prediction_index = data['skill_top_choice_index']

            # if top_choice not in sanity_check:
            #     sanity_check[top_choice] = {
            #         'tie': 0,
            #         'wrong': 0,
            #         'count': 0
            #     }
            
            # sanity_check[top_choice]['count'] += 1
            # if top_choice_index_from_claude == small_model_prediction_index:
            #     sanity_check[top_choice]['tie'] += 1
            # else:
            #     sanity_check[top_choice]['wrong'] += 1
            
            output = {
                'text': dataset[prompt_index]['text'],
                'url': dataset[prompt_index]['url'],
                'choice': top_choice,
                'label': int(top_choice_index),
            }
            json.dump(output, f)
            f.write('\n')
            
            # time.sleep(20)
        

    # print("\nSanity check results:")
    # with open('sanity_check.json', 'w') as f:
    #     json.dump(sanity_check, f, indent=2)
    

    
    # outputs = llm.generate(
    #     [prompts[0]], 
    #     sampling_params, 
    #     guided_options_request=dict(guided_choice=prompt_config.labels))

    # print(outputs)



    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    print(f"Total skill seen {len(skills_seen)}")
    print(skills_seen)