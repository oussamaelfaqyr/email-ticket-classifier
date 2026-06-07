import os
import yaml
import pickle
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_configs():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_baseline(features_dir: str, model_out: str):
    params = load_params()
    configs = load_configs()
    
    mlflow.set_tracking_uri(configs["mlflow"]["tracking_uri"])
    mlflow.set_experiment(configs["mlflow"]["experiment_name"])
    
    with open(os.path.join(features_dir, "train_features.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)
        
    with open(os.path.join(features_dir, "test_features.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)
        
    with mlflow.start_run():
        C = params["training"]["baseline"]["C"]
        max_iter = params["training"]["baseline"]["max_iter"]
        
        mlflow.log_params({"C": C, "max_iter": max_iter})
        
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=params["base"]["random_state"])
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_metrics({"f1_score": f1, "accuracy": acc})
        
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        with open(model_out, "wb") as f:
            pickle.dump(clf, f)
            
        mlflow.sklearn.log_model(clf, "model")
        print(f"Model trained. F1: {f1:.4f}, Accuracy: {acc:.4f}. Saved to {model_out}")

if __name__ == "__main__":
    train_baseline("data/processed/", "models/baseline.pkl")
