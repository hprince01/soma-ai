"""
Experiment: do character n-grams generalise better to NEW news websites (the official test split)?
Kinyarwanda is agglutinative (umu-kinnyi, aba-kinnyi, ...), so sub-word pieces may transfer across sources.
Run: .venv/bin/python experiment_char.py
"""
import json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from train import load, normalize, SEED

train, test = load("train"), load("test")
X = [normalize(r["text"]) for r in train]; y = np.array([r["y"] for r in train])
Xte = [normalize(r["text"]) for r in test]; yte = np.array([r["y"] for r in test])
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)

results = []
for ngr, mf in [((3, 5), 60000), ((2, 4), 40000)]:
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngr, min_df=3, max_features=mf, sublinear_tf=True)
    A = vec.fit_transform(Xtr); V = vec.transform(Xva)
    clf = LogisticRegression(C=5, class_weight="balanced", max_iter=2000, random_state=SEED).fit(A, ytr)
    pv = clf.predict(V)
    vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=ngr, min_df=3, max_features=mf, sublinear_tf=True)
    A2 = vec2.fit_transform(X); T2 = vec2.transform(Xte)
    clf2 = LogisticRegression(C=5, class_weight="balanced", max_iter=2000, random_state=SEED).fit(A2, y)
    pt = clf2.predict(T2)
    row = {"features": f"char_wb {ngr[0]}-{ngr[1]}, {mf}", "val_acc": round(accuracy_score(yva, pv), 4), "val_macro_f1": round(f1_score(yva, pv, average="macro"), 4),
           "test_acc": round(accuracy_score(yte, pt), 4), "test_macro_f1": round(f1_score(yte, pt, average="macro"), 4), "seconds": round(time.time() - t0)}
    print(row, flush=True); results.append(row)
json.dump(results, open("results/experiment_char.json", "w"), indent=1)
