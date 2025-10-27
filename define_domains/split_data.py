from sklearn.model_selection import train_test_split
import pandas as pd
import json



def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# data = load_jsonl("sanity_check_sample-10K_model-llama70B.jsonl")

df = pd.read_json("sanity_check_sample-10K_model-llama70B.jsonl", lines=True)

print(df.head())
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,  # 20% for temp (validation + test)
    stratify=df['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 50% of the temp_df for test (so 10% of original)
    random_state=42
)


print("len of train_df:", len(train_df))
print("len of val_df:", len(val_df))
print("len of test_df:", len(test_df))


train_df.to_json("sanity_check_sample-10K_model-llama70B-train.jsonl", orient='records', lines=True)
val_df.to_json("sanity_check_sample-10K_model-llama70B-val.jsonl", orient='records', lines=True)
test_df.to_json("sanity_check_sample-10K_model-llama70B-test.jsonl", orient='records', lines=True)