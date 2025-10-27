import json
import argparse
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_skill_prob_distribution(file_path: str):
    """
    Reads a jsonl file, extracts 'skill_top_choice_prob' values,
    and plots their distribution as a histogram.

    Args:
        file_path (str): The path to the jsonl file.
    """
    probabilities: List[float] = []
    lines_processed = 0
    lines_with_key = 0
    lines_failed_decode = 0
    label_count = {}

    print(f"Reading data from '{file_path}'...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines_processed += 1
                try:
                    data = json.loads(line)
                    if 'skill_top_choice_prob' in data:
                        prob = data['skill_top_choice_prob']
                        if isinstance(prob, (int, float)):
                            probabilities.append(float(prob))
                            lines_with_key += 1
                    
                    if "skill_top_choice_index" in data:
                        label_id = data['skill_top_choice_index']
                        if label_id not in label_count:
                            label_count[label_id] = 0
                        label_count[label_id] += 1
                    

                except json.JSONDecodeError:
                    lines_failed_decode += 1
                    print(f"Warning: Could not decode JSON from line {lines_processed}: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    print("\n--- Processing Summary ---")
    print(f"Total lines processed: {lines_processed}")
    print(f"Lines with 'skill_top_choice_prob' key: {lines_with_key}")
    if lines_failed_decode > 0:
        print(f"Lines that failed JSON decoding: {lines_failed_decode}")
    print("--------------------------\n")

    if not probabilities:
        print("No valid 'skill_top_choice_prob' values were found. Cannot generate a plot.")
        return

    # --- Calculate Statistics ---
    mean_prob = np.mean(probabilities)
    median_prob = np.median(probabilities)
    std_dev = np.std(probabilities)
    
    print("--- Statistics for 'skill_top_choice_prob' ---")
    print(f"  - Mean:   {mean_prob:.4f}")
    print(f"  - Median: {median_prob:.4f}")
    print(f"  - Std Dev:{std_dev:.4f}")
    print(f"  - Min:    {min(probabilities):.4f}")
    print(f"  - Max:    {max(probabilities):.4f}")
    print("--------------------------------------------\n")

    # --- Plotting the distribution ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.hist(probabilities, bins=50, edgecolor='black', alpha=0.75, density=True)
    
    plt.axvline(mean_prob, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_prob:.2f}')
    plt.axvline(median_prob, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_prob:.2f}')

    plt.title('Distribution of skill_top_choice_prob', fontsize=18, pad=20)
    plt.xlabel('Probability', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    
    print("Displaying plot. Close the plot window to exit the script.")
    plt.savefig("skill_top_choice_prob_distribution.png")
    plt.show()

    total_count = sum(label_count.values())
    label_percentage = {label: (count / total_count) * 100 for label, count in label_count.items()}

    print("Label Distribution (in percentage):")
    for label, percentage in label_percentage.items():
        print(f"Label {label}: {percentage:.4f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the distribution of 'skill_top_choice_prob' from a JSONL file.")
    parser.add_argument("file_path", type=str, help="Path to the .jsonl training data file.")
    args = parser.parse_args()
    plot_skill_prob_distribution(args.file_path)