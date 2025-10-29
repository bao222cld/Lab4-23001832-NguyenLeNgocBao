# src/models/vectorizers.py
import re
from typing import List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SKTfidf
from gensim.models import Word2Vec

class RegexTokenizer:
    def __init__(self):
        self.pattern = re.compile(r"\b[a-zA-Z0-9']+\b")

    def __call__(self, text: str):
        return self.pattern.findall(text.lower() if isinstance(text, str) else "")

class TfidfVectorizerWrapper:
    def __init__(self, tokenizer: Optional[Any]=None, max_features:int=10000, ngram_range=(1,2)):
        if tokenizer is None:
            tokenizer = RegexTokenizer()
        self.tokenizer = tokenizer
        self.v = SKTfidf(tokenizer=self.tokenizer, max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, texts: List[str]):
        return self.v.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.v.transform(texts)

class Word2VecAvgVectorizer:
    def __init__(self, tokenizer: Optional[Any]=None, vector_size:int=100, min_count:int=1, window:int=5, workers:int=2, seed:int=42):
        if tokenizer is None:
            tokenizer = RegexTokenizer()
        self.tokenizer = tokenizer
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        self.seed = seed
        self.model: Optional[Word2Vec] = None

    def fit_transform(self, texts: List[str]):
        tokenized = [self.tokenizer(t) for t in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size,
                              min_count=self.min_count, window=self.window,
                              workers=self.workers, seed=self.seed)
        vectors = [self._avg_vector(tokens) for tokens in tokenized]
        return np.vstack(vectors)

    def transform(self, texts: List[str]):
        if self.model is None:
            raise RuntimeError("Word2Vec model not trained. Call fit_transform first.")
        tokenized = [self.tokenizer(t) for t in texts]
        vectors = [self._avg_vector(tokens) for tokens in tokenized]
        return np.vstack(vectors)

    def _avg_vector(self, tokens):
        vs = []
        for t in tokens:
            if t in self.model.wv:
                vs.append(self.model.wv[t])
        if len(vs) == 0:
            return np.zeros(self.vector_size)
        return np.mean(vs, axis=0)