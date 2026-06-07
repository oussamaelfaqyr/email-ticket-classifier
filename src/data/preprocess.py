import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess(input_path: str, train_out: str, test_out: str):
    params = load_params()
    df = pd.read_csv(input_path)
    
    # Combine subject and text for classification
    if "subject" in df.columns:
        df["text"] = df["subject"].fillna("") + " " + df["text"].fillna("")
        
    df["text"] = df["text"].str.lower().str.strip()
    df = df.dropna(subset=["label", "text"])
    
    train_df, test_df = train_test_split(
        df, 
        test_size=params["data"]["test_size"], 
        random_state=params["base"]["random_state"],
        stratify=df["label"]
    )
    
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Preprocessed data: {len(train_df)} train, {len(test_df)} test.")

if __name__ == "__main__":
    preprocess("data/raw/data.csv", "data/processed/train.csv", "data/processed/test.csv")
