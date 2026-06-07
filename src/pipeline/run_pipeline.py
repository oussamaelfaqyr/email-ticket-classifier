import os
from src.data.ingest import ingest_data
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.training.baseline.train import train_baseline
from src.training.bert.train import train_bert
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "bert"], help="Model to train")
    args = parser.parse_args()

    print("=== Pipeline Started ===")
    
    raw_data_path = "data/raw/data.csv"
    train_data_path = "data/processed/train.csv"
    test_data_path = "data/processed/test.csv"
    features_dir = "data/processed/"
    baseline_path = "models/baseline.pkl"
    bert_path = "models/bert_model"
    
    print("\n1. Data Ingestion...")
    ingest_data("data/raw/data.csv", raw_data_path)
    
    print("\n2. Data Preprocessing...")
    preprocess(raw_data_path, train_data_path, test_data_path)
    
    if args.model == "baseline":
        print("\n3. Feature Extraction (TF-IDF)...")
        build_features(train_data_path, test_data_path, features_dir)
        
        print("\n4. Model Training (Baseline)...")
        train_baseline(features_dir, baseline_path)
    else:
        print("\n3. Model Training (DistilBERT)...")
        train_bert(train_data_path, test_data_path, bert_path)
        
    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    main()
