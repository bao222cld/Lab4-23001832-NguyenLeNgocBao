# experiments/run_experiments.py
import os, json
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from src.models.vectorizers import TfidfVectorizerWrapper, Word2VecAvgVectorizer
from src.models.text_classifier import TextClassifier
from datasets import load_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset_hf(sample_size:int=None):
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
        df = ds['train'].to_pandas()
        # heuristics to find text and label
        text_col = 'text' if 'text' in df.columns else next((c for c in df.columns if df[c].dtype == object), df.columns[0])
        label_col = next((c for c in df.columns if 'label' in c or 'sentiment' in c), None)
        if label_col is None:
            for c in df.columns:
                if df[c].dtype in ('int64','float64') and df[c].nunique() < 10:
                    label_col = c
                    break
        if label_col is None:
            raise ValueError("Cannot find label column automatically.")
        df = df[[text_col, label_col]].dropna().rename(columns={text_col: 'text', label_col: 'label'})
        if df['label'].dtype.kind in 'iuf' and set(df['label'].unique()) <= {-1,0,1}:
            df['label'] = df['label'].apply(lambda x: 1 if x>0 else 0)
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        return df
    except Exception as e:
        print("HF load failed:", e)
        return None

def run(sample_size:int = 2000):
    df = load_dataset_hf(sample_size=sample_size)
    if df is None:
        # fallback tiny dataset
        data = [
            ("I love this product, it's great", 1),
            ("Terrible, worst ever", 0),
            ("Amazing quality", 1),
            ("Do not buy this", 0),
            ("Highly recommended", 1),
            ("Not good at all", 0),
        ]
        df = pd.DataFrame(data, columns=['text','label'])

    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels))>1 else None)

    results = []

    # Baseline: TF-IDF + LogisticRegression
    tfidf = TfidfVectorizerWrapper()
    clf = TextClassifier(tfidf)
    clf.fit(X_tr, y_tr, algorithm='logreg', class_weight='balanced')
    preds = clf.predict(X_te)
    m = clf.evaluate(y_te, preds)
    m.update({"model":"tfidf_logreg_baseline"})
    results.append(m)
    print("Baseline:", m)

    # TF-IDF + MultinomialNB
    clf_nb = TextClassifier(tfidf)
    clf_nb.fit(X_tr, y_tr, algorithm='nb')
    preds_nb = clf_nb.predict(X_te)
    mnb = clf_nb.evaluate(y_te, preds_nb)
    mnb.update({"model":"tfidf_multinb"})
    results.append(mnb)
    print("NB:", mnb)

    # TF-IDF + LogReg with GridSearch
    tfidf2 = TfidfVectorizerWrapper(max_features=10000)
    clf_gs = TextClassifier(tfidf2)
    param_grid = {'C':[0.01,0.1,1,10]}
    clf_gs.fit(X_tr, y_tr, algorithm='logreg', do_grid_search=True, grid_params=param_grid, class_weight='balanced')
    preds_gs = clf_gs.predict(X_te)
    m_gs = clf_gs.evaluate(y_te, preds_gs)
    m_gs.update({"model":"tfidf_logreg_gridsearch", "best_params": getattr(clf_gs,'best_params_',None)})
    results.append(m_gs)
    print("GridSearch:", m_gs)

    # Word2Vec-average + LogReg
    w2v = Word2VecAvgVectorizer()
    clf_w2v = TextClassifier(w2v)
    clf_w2v.fit(X_tr, y_tr, algorithm='logreg')
    preds_w2v = clf_w2v.predict(X_te)
    m_w2v = clf_w2v.evaluate(y_te, preds_w2v)
    m_w2v.update({"model":"word2vec_avg_logreg"})
    results.append(m_w2v)
    print("Word2Vec:", m_w2v)

    # Save results
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print("Saved results to outputs/")
    return results

if __name__ == "__main__":
    run(sample_size=2000)