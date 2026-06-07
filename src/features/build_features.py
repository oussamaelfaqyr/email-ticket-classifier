import pandas as pd
import os
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def build_features(train_path: str, test_path: str, out_dir: str):
    params = load_params()
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    vectorizer = TfidfVectorizer(
        max_features=params["features"]["max_features"],
        ngram_range=tuple(params["features"]["ngram_range"])
    )
    
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(out_dir, "train_features.pkl"), "wb") as f:
        pickle.dump((X_train, train_df["label"]), f)
        
    with open(os.path.join(out_dir, "test_features.pkl"), "wb") as f:
        pickle.dump((X_test, test_df["label"]), f)
        
    print(f"Features built and saved to {out_dir}")

if __name__ == "__main__":
    build_features("data/processed/train.csv", "data/processed/test.csv", "data/processed/")
