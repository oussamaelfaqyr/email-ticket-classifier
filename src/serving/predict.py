from typing import Tuple
from src.serving.model_loader import ModelLoader

class Predictor:
    def __init__(self, loader: ModelLoader):
        self.loader = loader
        
    def predict(self, text: str) -> Tuple[str, float]:
        if not self.loader.is_loaded():
            self.loader.load()
            
        X = self.loader.vectorizer.transform([text])
        probabilities = self.loader.model.predict_proba(X)[0]
        
        best_class_idx = probabilities.argmax()
        label = self.loader.model.classes_[best_class_idx]
        confidence = probabilities[best_class_idx]
        
        return label, float(confidence)
