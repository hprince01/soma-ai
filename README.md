# ☀️ Soma AI — Kinyarwanda News Topic Classifier

**Live demo → [princehirwa.pages.dev/projects/soma-ai](https://princehirwa.pages.dev/projects/soma-ai/)**

*Soma* means **“read”** in Kinyarwanda. Soma AI reads Kinyarwanda news text and predicts its topic (politics, sport, economy, health, …), highlighting the words that drove each decision. It was trained in Python on **21,268 real articles from 20 Rwandan news sources**, and the trained model runs **entirely in the browser** in plain JavaScript: no server, and user text never leaves the device.

> *Soma AI: Ireba icyo inkuru ivuga.* (It sees what the story is about.)

## Results

| Setting | Accuracy | Macro F1 |
|---|---|---|
| New articles from **news sites seen in training** (15 % validation split) | **89.5 %** | 81.1 % |
| Official KINNEWS test set: **news sites never seen in training** | **69.1 %** | 53.9 % |

### The key finding: domain shift
The official test split comes from **different websites** (mostly igihe.com and kigalipost.com) than the training data (mostly kigalitoday.com, muhabura.rw, ukwezi.rw). Each newsroom writes and labels topics differently (one site’s *culture* is another’s *history*, environment stories are often filed as *economy*), so accuracy drops about 20 points on unseen sources. Topics with distinctive vocabulary transfer well (sport F1 0.93, relationship 0.79, entertainment 0.78, religion 0.74); topics with few training examples or inconsistent labels do not (education 0.12, history 0.11, culture 0.28, environment 0.30).

### Model comparison (selected on validation, scored on the official test set)

| Model | Features | Val acc. | Test acc. | Test macro F1 |
|---|---|---|---|---|
| Complement Naive Bayes | words, 40k | 86.7 % | 67.7 % | 42.8 % |
| Linear SVM | words + pairs, 20k | 90.0 % | 71.0 % | 54.7 % |
| **Logistic Regression (C=5, balanced) — live demo** | words + pairs, 20k | 89.5 % | 69.1 % | 53.9 % |
| Logistic Regression — experiment | character 3–5-grams, 60k | 89.8 % | **71.7 %** | **57.4 %** |

Character n-grams generalise slightly better to new sites, since Kinyarwanda builds words from prefixes and roots (*umu-*, *aba-*, *-kinnyi*). The live demo still uses the word model because it **explains itself with whole Kinyarwanda words** and is half the size, a deliberate accuracy-vs-explainability trade-off.

## How it works
1. **Data:** KINNEWS raw split (17,014 train / 4,254 test, 14 classes)
2. **Preprocess:** lowercase, Unicode NFKD with accents removed, tokens `[a-z]{2,}`
3. **Features:** TF-IDF over words and word pairs (sublinear TF, L2 norm, min_df=2, top 20,000 terms)
4. **Model:** multinomial logistic regression with balanced class weights, chosen by macro-F1 on a stratified validation split (the official test set is used once, at the end)
5. **Export:** vocabulary, IDF and **int8-quantised** weights in a ~750 KB JSON file; quantisation costs 0.1 points of accuracy
6. **Browser inference:** the same pipeline in vanilla JS. **Verified: identical topic on 150/150 test articles, max probability difference 6×10⁻⁶**
7. **Explainability:** each word’s contribution (weight × TF-IDF) towards the predicted class is shown in green or red

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./get_data.sh                       # downloads KINNEWS (MIT licence)
.venv/bin/python train.py           # trains, evaluates, exports model.json + metrics
.venv/bin/python experiment_char.py # optional character n-gram experiment
```
Outputs: `results/metrics.json` (all scores, per-class report, confusion matrix), `results/experiment_char.json`, and the web model in `../../portfolio/projects/soma-ai/`.

## Limitations
- Learns topic **vocabulary**, not meaning; it can be fooled by mixed-topic stories.
- Very imbalanced classes (environment: 38 training articles, fashion: 54).
- Trained on news from around 2020; new names and events may be unfamiliar.
- Labels come from each website’s own categories, which are not always consistent.

**Next steps:** fine-tune an African multilingual transformer (e.g. AfroXLMR / AfriBERTa) and compare it against this lightweight, explainable baseline; try domain-adaptation techniques for unseen news sources.

## Data & credits
**KINNEWS**: R. A. Niyongabo, H. Qu, J. Kreutzer, L. Huang. *KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi.* COLING 2020. [arXiv:2010.12174](https://arxiv.org/abs/2010.12174) · MIT licence. Article copyrights belong to the original news sources; this repository does **not** redistribute the articles (use `get_data.sh`).

---
Built by **Prince Hirwa Rutagengwa · [Kai Labs](https://princehirwa.pages.dev/)**, Kigali, Rwanda.
