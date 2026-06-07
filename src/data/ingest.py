import os
import shutil

def ingest_data(input_path: str, output_path: str):
    """
    In a real scenario, this would download from Kaggle using their API.
    For now, we just copy the file if it exists at the source.
    """
    if input_path != output_path and os.path.exists(input_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"Ingested data into {output_path}")
    elif os.path.exists(output_path):
        print(f"Data already present at {output_path}")
    else:
        print(f"Warning: Data not found at {input_path} or {output_path}")

if __name__ == "__main__":
    # We expect the file to already be at data/raw/data.csv via manual copy
    ingest_data("data/raw/data.csv", "data/raw/data.csv")
