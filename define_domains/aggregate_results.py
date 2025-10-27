import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def load_job_jsonl_files(directory: Path) -> List[Dict[str, Any]]:
    """
    Loads all JSONL files in a directory matching the pattern '*-job{0-23}.jsonl'.

    The function first finds all '.jsonl' files in the given directory. It then
    filters them using a regular expression to find files that end with
    '-job<number>.jsonl', where the number is between 0 and 23, inclusive.

    Args:
        directory: The path to the directory containing the files.

    Returns:
        A list containing all the parsed JSON objects from the matching files.

    Raises:
        NotADirectoryError: If the provided path is not a valid directory.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"The path '{directory}' is not a valid directory.")

    # Regex to match '-job' followed by a number and '.jsonl' at the end of the filename.
    file_pattern = re.compile(r"-job(\d+)\.jsonl$")
    all_data = []

    # Use glob to find potential files, which is efficient.
    # Sorting ensures a consistent order, though not strictly necessary.
    candidate_files = sorted(directory.glob('*.jsonl'))

    print(f"Searching in directory: {directory}")
    print(f"Found {len(candidate_files)} potential .jsonl files to check.")

    for file_path in candidate_files:
        print(f"Processing file: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            seen_page = set()
            counter = 0
            duplicate = 0
            for line in f:
                try:
                    data = json.loads(line)
                    if data['metadata']["WARC-Record-ID"] not in seen_page:
                        seen_page.add(data['metadata']["WARC-Record-ID"])
                        all_data.append(data)
                        counter += 1
                    else:
                        duplicate += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode a JSON line in {file_path.name}")
            
            print(f"file: {file_path.name}, num_of_valid_lines: {counter}, num_of_duplicate_lines: {duplicate}")

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load all JSONL files from a directory that end with '-job{0-23}.jsonl'."
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        help="The path to the directory containing the JSONL files.",
        default="datasets",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        choices=["train", "test", "validation"],
        default="train")


    args = parser.parse_args()
    data_type = args.data_type
    input_dir = Path(f"{args.input_directory}/{data_type}")

    loaded_data = load_job_jsonl_files(input_dir)
    print(f"\nSuccessfully loaded a total of {len(loaded_data):,} records.")
    
    with open(f"{args.input_directory}/{data_type}_dclm_refinedweb_1M_Qwen3-14B.jsonl", 'w') as f:
        for line in loaded_data:
            json.dump(line, f)
            f.write("\n")
    
    print("Done")