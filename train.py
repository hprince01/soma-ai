"""
Soma AI — Kinyarwanda news topic classifier
===========================================
Trains and compares text classifiers on KINNEWS (Niyongabo et al., COLING 2020; MIT licence),
evaluates on the official held-out test split, and exports the chosen model as a compact JSON
file that the browser demo runs with plain JavaScript (no server).

Run:  .venv/bin/python train.py
"""
import csv, json, math, re, time, unicodedata, base64, random
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "KINNEWS" / "raw"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
WEB = ROOT.parent.parent / "portfolio" / "projects" / "soma-ai"; WEB.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

EN = ["politics","sport","economy","health","entertainment","history","technology","tourism",
      "culture","fashion","religion","environment","education","relationship"]

# ---------------------------------------------------------------- data
def load(split):
    csv.field_size_limit(10**9)
    rows = []
    with open(RAW / f"{split}.csv", encoding="utf-8") as f:
        r = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(r)
        for label, kin, en, url, title, content in r:
            rows.append({"y": int(label) - 1, "kin": kin, "url": url, "title": title, "text": f"{title} {content}"})
    return rows

# ---------------------------------------------------------------- preprocessing (mirrored exactly in JS)
TOKEN = re.compile(r"[a-z]{2,}")
def normalize(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))
def tokenize(s):
    return TOKEN.findall(normalize(s))

def vectorizer(ngram, max_features):
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=ngram,
                           min_df=2, max_features=max_features, sublinear_tf=True, norm="l2", smooth_idf=True)

def source_counts(rows):
    c = {}
    for r in rows:
        h = re.sub(r"^https?://(www\.)?", "", r["url"]).split("/")[0].rstrip(".")
        c[h] = c.get(h, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1])[:8])

def main():
    t0 = time.time()
    train, test = load("train"), load("test")
    kin_names = {}
    for r in train + test: kin_names.setdefault(r["y"], r["kin"])
    Xtr_all = [r["text"] for r in train]; ytr_all = np.array([r["y"] for r in train])
    Xte = [r["text"] for r in test];       yte = np.array([r["y"] for r in test])
    print(f"train {len(train)}  test {len(test)}")

    # validation split (from TRAIN only) for choosing settings — the official test set stays untouched
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.15, random_state=SEED, stratify=ytr_all)

    candidates = []
    for ngram in [(1, 1), (1, 2)]:
        for mf in [20000, 40000]:
            vec = vectorizer(ngram, mf); A = vec.fit_transform(Xtr); V = vec.transform(Xva)
            models = {
                "Complement Naive Bayes": ComplementNB(alpha=0.3),
                "Linear SVM": LinearSVC(C=0.5, class_weight="balanced", random_state=SEED),
            }
            for C in [5, 20]:
                for cw in [None, "balanced"]:
                    models[f"Logistic Regression (C={C}{', balanced' if cw else ''})"] = LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=SEED)
            for name, m in models.items():
                m.fit(A, ytr); p = m.predict(V)
                row = {"model": name, "ngram": list(ngram), "max_features": mf,
                       "val_acc": accuracy_score(yva, p), "val_macro_f1": f1_score(yva, p, average="macro")}
                candidates.append(row)
                print(f"{name:42s} ngram={ngram} mf={mf:6d}  acc={row['val_acc']:.3f}  macroF1={row['val_macro_f1']:.3f}")

    # the browser needs probabilities + per-word explanations, so the web model is the best LOGISTIC REGRESSION
    lr = [c for c in candidates if c["model"].startswith("Logistic") and c["max_features"] == 20000]  # 20k keeps the web model small
    best = max(lr, key=lambda c: (c["val_macro_f1"], c["val_acc"]))
    print("\nselected for web:", best)

    # final fit on ALL training data, evaluate once on the official test set
    C = 5 if "C=5" in best["model"] else 20
    cw = "balanced" if "balanced" in best["model"] else None
    vec = vectorizer(tuple(best["ngram"]), best["max_features"])
    A = vec.fit_transform(Xtr_all); T = vec.transform(Xte)
    clf = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=SEED).fit(A, ytr_all)
    pred = clf.predict(T)

    # also score the best baseline of each family on the test set, for an honest comparison table
    comparison = []
    for fam in ["Complement Naive Bayes", "Linear SVM", "Logistic Regression"]:
        c = max([x for x in candidates if x["model"].startswith(fam)], key=lambda x: x["val_macro_f1"])
        v2 = vectorizer(tuple(c["ngram"]), c["max_features"]); A2 = v2.fit_transform(Xtr_all); T2 = v2.transform(Xte)
        if fam.startswith("Complement"): m = ComplementNB(alpha=0.3)
        elif fam.startswith("Linear"): m = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
        else: m = LogisticRegression(C=5 if "C=5" in c["model"] else 20, class_weight="balanced" if "balanced" in c["model"] else None, max_iter=3000, random_state=SEED)
        m.fit(A2, ytr_all); p2 = m.predict(T2)
        comparison.append({"model": c["model"], "ngram": c["ngram"], "max_features": c["max_features"],
                           "val_macro_f1": round(float(c["val_macro_f1"]), 4), "val_acc": round(c["val_acc"], 4),
                           "test_acc": round(accuracy_score(yte, p2), 4), "test_macro_f1": round(f1_score(yte, p2, average="macro"), 4)})
    print("\ncomparison on test:", json.dumps(comparison, indent=1))

    rep = classification_report(yte, pred, target_names=EN, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred).tolist()
    metrics = {
        "model": best["model"], "ngram": best["ngram"], "max_features": best["max_features"],
        "test_accuracy": round(accuracy_score(yte, pred), 4),
        "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "test_weighted_f1": round(f1_score(yte, pred, average="weighted"), 4),
        "per_class": {EN[i]: {k: round(rep[EN[i]][k], 4) for k in ["precision", "recall", "f1-score", "support"]} for i in range(14)},
        "confusion_matrix": cm, "comparison": comparison, "validation_runs": candidates,
        "train_counts": np.bincount(ytr_all, minlength=14).tolist(), "test_counts": np.bincount(yte, minlength=14).tolist(),
        "n_train": len(train), "n_test": len(test), "seconds": round(time.time() - t0, 1),
        "val_accuracy": round(best["val_acc"], 4), "val_macro_f1": round(float(best["val_macro_f1"]), 4),
        "sources": {"train": source_counts(train), "test": source_counts(test)},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\nTEST accuracy {metrics['test_accuracy']}  macro-F1 {metrics['test_macro_f1']}")

    # ------------------------------------------------------------ export for the browser
    vocab = vec.get_feature_names_out().tolist()
    W = clf.coef_.astype(np.float64)            # (14, F)
    scale = np.abs(W).max(axis=1) / 127.0       # per-class int8 quantisation
    Q = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
    idf = np.round(vec.idf_, 4).tolist()
    model = {
        "name": "Soma AI", "version": 1,
        "classes": [{"en": EN[i], "kin": kin_names[i]} for i in range(14)],
        "preprocess": {"normalize": "lowercase + NFKD, strip combining marks", "token": "[a-z]{2,}", "ngram": best["ngram"],
                       "tf": "1+log(count)", "norm": "l2"},
        "vocab": vocab, "idf": idf,
        "coef_int8_b64": base64.b64encode(Q.tobytes()).decode(), "coef_scale": scale.round(8).tolist(),
        "intercept": clf.intercept_.round(6).tolist(),
        "metrics": {k: metrics[k] for k in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "per_class", "confusion_matrix",
                                            "comparison", "train_counts", "test_counts", "n_train", "n_test", "model",
                                            "val_accuracy", "val_macro_f1", "sources"]},
    }
    (WEB / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    # ------------------------------------------------------------ parity check data (quantised model, Python reference)
    Wq = Q.astype(np.float64) * scale[:, None]
    def predict_q(texts):
        X = vec.transform(texts)
        z = X @ Wq.T + clf.intercept_
        z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)
    idx = random.sample(range(len(test)), 150)
    probs = predict_q([test[i]["text"] for i in idx])
    parity = [{"text": test[i]["text"], "probs": probs[j].round(6).tolist()} for j, i in enumerate(idx)]
    (OUT / "parity.json").write_text(json.dumps(parity))
    qacc = accuracy_score(yte, predict_q(Xte).argmax(1))
    print(f"quantised model test accuracy {qacc:.4f} (vs float {metrics['test_accuracy']})")

    # ------------------------------------------------------------ short headline examples (title only, credited)
    by_cls = {}
    for r in test:
        if len(r["title"]) < 110 and not re.search(r"1994|jenoside|genocide|abatutsi|biciwe", r["title"], re.I): by_cls.setdefault(r["y"], []).append(r)
    examples = []
    for y in range(14):
        pool = by_cls.get(y, [])
        random.shuffle(pool)
        for r in pool[:30]:
            p = predict_q([r["title"]])[0]
            if p.argmax() == y:
                host = re.sub(r"^https?://(www\.)?", "", r["url"]).split("/")[0]
                examples.append({"title": r["title"].strip(), "label": EN[y], "source": host, "url": r["url"]}); break
    (WEB / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))
    print(f"examples: {len(examples)}  web model size: {(WEB/'model.json').stat().st_size/1024:.0f} KB  total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
