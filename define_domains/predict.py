import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import numpy as np
import os


def predict(model, tokenizer, dataset, batch_size=32, device="cuda"):
    """
    Perform inference on a dataset using a fine-tuned model.

    Args:
        model: The fine-tuned sequence classification model.
        tokenizer: The tokenizer for the model.
        dataset: The Hugging Face dataset to perform inference on.
        text_column (str): The name of the column containing the text.
        batch_size (int): The batch size for inference.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        A list of prediction dictionaries.
    """
    model.to(device)
    model.eval()

    all_predictions = []
    predictions_label_id = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        batch = dataset[i:i+batch_size]

        # texts = batch["text"]

        # Note: Some models, like the ones in this repository, might require a
        # specific input format (e.g., including a URL). You may need to
        # adjust how `texts` is constructed. For example:
        # {url}\n\n{text}
        
        texts = [f"{batch['url'][i]}\n\n{batch['text'][i]}" for i in range(len(batch['text']))]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192, # A reasonable default max_length
            return_tensors="pt"
        )

        # Move tensors to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_ids = torch.argmax(logits, dim=-1)

        predicted_class_ids = predicted_class_ids.cpu().numpy()

        for j in range(len(batch['text'])):
            predicted_class_id = int(predicted_class_ids[j])
            batch["metadata"][j]['WARC-Date'] = str(batch["metadata"][j]['WARC-Date'])

            doc = {
                "url": batch["url"][j],
                "text": batch["text"][j],
                "previous_word_count": batch["previous_word_count"][j],
                "metadata": batch["metadata"][j],
                "language_id_whole_page_fasttext": batch["language_id_whole_page_fasttext"][j],
                "predicted_label_id": predicted_class_id,
            }
            
            all_predictions.append(doc)
            predictions_label_id.append(predicted_class_id)        

    return all_predictions, predictions_label_id

def load_huggingface_dataset(dataset_name: str, file_number: int):
    """
    Load a dataset from the Hugging Face Hub.
    """
    id = str(file_number).zfill(4)
    files_to_download = [f"documents/CC_shard_0000{id}_processed.jsonl.zst"]
    dataset = load_dataset(
        dataset_name,
        data_files=files_to_download,
        split="train",
    )
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned sequence classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model or Hugging Face Hub repo ID.")
    parser.add_argument("--dataset_name", type=str, default="WebOrganizer/Corpus-200B", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--job_id", type=int, help="Job ID.", default=0)
    parser.add_argument("--num_jobs", default=24, type=int, help="Total number of jobs.")
    parser.add_argument("--output_dir", type=str, default="/data/group_data/cx_group/WebOrganizer/Corpus-30B", help="File to save the predictions.")
    args = parser.parse_args()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True)
    print("Model and tokenizer loaded.")

    total_file = 3_000
    start_file_id = 1 if args.job_id == 0 else args.job_id * (total_file // args.num_jobs)
    end_file_id = (args.job_id + 1) * (total_file // args.num_jobs) if args.job_id < args.num_jobs - 1 else total_file

    print(f"Processing files from {start_file_id} to {end_file_id}")
    
    for file_id in range(start_file_id, end_file_id):
        # check if this file exit
        id = str(file_id).zfill(4)
        npy_file = f"{args.output_dir}/predictions_label_id/CC_shard_0000{id}_processed_skill.npy"
        output_file = f"{args.output_dir}/documents/CC_shard_0000{id}_processed_skill.jsonl"
        if os.path.exists(npy_file) and os.path.exists(output_file):
            print(f"File already exist. {output_file}")
            continue

        try:
            dataset = load_huggingface_dataset(args.dataset_name, file_id)
        except:
            print(f"Error loading dataset. Skipping file id: {file_id}...")
            continue


        # Perform prediction
        predictions, predictions_label_id = predict(model, tokenizer, dataset, args.batch_size, device)
        
        # Save the predictions to a file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"\nPredictions saved to {output_file}")

        os.makedirs(f"{args.output_dir}/predictions_label_id", exist_ok=True)
        np.save(npy_file, predictions_label_id)



    

if __name__ == "__main__":
    main()