# src/models/text_classifier.py
import joblib
from typing import List, Any, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class TextClassifier:
    def __init__(self, vectorizer: Any):
        self.vectorizer = vectorizer
        self.model = None
        self.model_name = None
        self.is_fitted = False
        self.best_params_ = None

    def _make_classifier(self, algorithm: str = "logreg", class_weight: Optional[dict]=None, random_state: int=42):
        if algorithm == "logreg":
            return LogisticRegression(solver='liblinear', max_iter=1000, class_weight=class_weight, random_state=random_state)
        elif algorithm == "nb":
            return MultinomialNB()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def fit(self, texts: List[str], labels: List[int], algorithm: str = "logreg",
            do_grid_search: bool = False, grid_params: Optional[dict] = None,
            class_weight: Optional[dict] = None, random_state: int = 42):
        X = self.vectorizer.fit_transform(texts)
        clf = self._make_classifier(algorithm=algorithm, class_weight=class_weight, random_state=random_state)
        if do_grid_search and grid_params:
            gs = GridSearchCV(clf, grid_params, cv=3, scoring='f1_macro', n_jobs=-1)
            gs.fit(X, labels)
            self.model = gs.best_estimator_
            self.best_params_ = gs.best_params_
        else:
            self.model = clf
            self.model.fit(X, labels)
        self.model_name = algorithm
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit(...) first.")
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        return preds.tolist()

    def predict_proba(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if hasattr(self.model, "predict_proba"):
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)
        else:
            return None

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        average = 'binary' if len(set(y_true)) == 2 else 'macro'
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    def classification_report(self, y_true: List[int], y_pred: List[int]) -> str:
        return classification_report(y_true, y_pred, zero_division=0)

    def save(self, path: str):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "model_name": self.model_name,
            "best_params": getattr(self, "best_params_", None)
        }, path)

    @classmethod
    def load(cls, path: str):
        d = joblib.load(path)
        obj = cls(d["vectorizer"])
        obj.model = d["model"]
        obj.model_name = d.get("model_name")
        obj.best_params_ = d.get("best_params")
        obj.is_fitted = True
        return obj