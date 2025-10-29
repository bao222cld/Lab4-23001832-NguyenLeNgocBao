# test/lab5_test.py
from src.models.vectorizers import TfidfVectorizerWrapper
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split

def test_textclassifier_baseline():
    texts = [
        "I like this", "I hate this", "very good", "very bad",
        "awesome product", "terrible service"
    ]
    labels = [1,0,1,0,1,0]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
    vec = TfidfVectorizerWrapper()
    clf = TextClassifier(vec)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert isinstance(preds, list)
    assert len(preds) == 3
