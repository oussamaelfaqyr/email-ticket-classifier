import pickle
import os

class ModelLoader:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")
            
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None
