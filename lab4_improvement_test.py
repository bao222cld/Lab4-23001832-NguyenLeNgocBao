# test/lab5_improvement_test.py
from src.models.vectorizers import TfidfVectorizerWrapper
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split

def test_improvement_nb_not_worse():
    texts = [
        "I love it", "So good", "This is great", "I hate it", "So bad", "Terrible"
    ]
    labels = [1,1,1,0,0,0]
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
    vec = TfidfVectorizerWrapper()
    base = TextClassifier(vec)
    base.fit(Xtr, ytr, algorithm='logreg')
    base_preds = base.predict(Xte)
    base_m = base.evaluate(yte, base_preds)
    nb = TextClassifier(vec)
    nb.fit(Xtr, ytr, algorithm='nb')
    nb_preds = nb.predict(Xte)
    nb_m = nb.evaluate(yte, nb_preds)
    tiny_eps = 1e-6
    assert nb_m['f1'] + tiny_eps >= base_m['f1']
