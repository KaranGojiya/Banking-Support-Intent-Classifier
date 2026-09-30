# Banking Support Intent Classifier

An NLP web app that reads a customer's banking message, predicts what the customer needs from **77 possible intents**, and suggests which support team should handle it.

**[Live Demo](https://banking-support-intent-classifier.streamlit.app/)**

## Features

- Predicts the customer's intent from free text, for example "Why was I charged twice?" becomes *Transaction charged twice*
- Shows the confidence score and a High / Medium / Low confidence level
- Shows the five next most likely intents with their confidence
- Suggests a support team for all 77 intents
- Highlights the keywords in the message that the model recognizes
- Warns when confidence is low so the query can go to a human agent
- One-click example queries and a history of the last 10 analyzed queries
- Model performance shown inside the app

## Model

A scikit-learn `Pipeline` with two steps:

| Step | Details |
|---|---|
| `TfidfVectorizer` | Single words, lowercase, `min_df=2`, `max_df=0.9`, `sublinear_tf=True`, 1,459-word vocabulary |
| `MultinomialNB` | `alpha=0.1` |

Trained on the **Banking77** dataset (PolyAI): customer queries labeled with 77 banking intents. The data file used here has 13,069 queries, split into 9,993 for training and 3,076 for testing.

## Results

Measured on the 3,076 test queries.

| Metric | Result |
|---|---|
| Accuracy | 93.60% |
| Macro F1-score | 93.59% |
| Macro precision | 93.76% |
| Macro recall | 93.59% |
| Top-3 accuracy | 98.44% |
| Top-5 accuracy | 99.35% |

Top-3 accuracy means the correct intent is among the model's three highest-ranked guesses.

### Confidence vs accuracy

Higher confidence really does mean the prediction is more reliable, so the app uses these bands:

| Confidence | Share of test queries | Correct predictions |
|---|---|---|
| 50% or more (High) | 77.0% | 98.6% |
| 30% to 50% (Medium) | 15.7% | 84.5% |
| Below 30% (Low) | 7.3% | 59.8% |

The weakest intents are transfer-related ones such as *pending transfer* (F1 0.76), *transfer not received by recipient* (0.80) and *balance not updated after bank transfer* (0.82), because their wording overlaps heavily.

## Tech stack

Python, scikit-learn, Pandas, NumPy, Joblib, Streamlit

## Project structure

```
Banking-Support-Intent-Classifier/
├── app.py                          # Streamlit app
├── banking_intent_classifier.pkl   # Trained TF-IDF + Naive Bayes pipeline
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/KaranGojiya/-Banking-Support-Intent-Classifier.git
cd -Banking-Support-Intent-Classifier
pip install -r requirements.txt
streamlit run app.py
```

The model was saved with scikit-learn 1.6.1, so `requirements.txt` pins that version to avoid loading warnings.

## Limitations

- English only, and each message is analyzed on its own.
- The team suggestions come from a hand-written mapping and are a demo of routing, not a real bank's rules.
- Naive Bayes confidence is spread thin across 77 classes, so short or vague messages often get low confidence even when the top guess is right.
- The model only knows the 77 Banking77 intents. Anything outside banking support will still be forced into one of them.

## Future improvements

- Compare against Logistic Regression and Linear SVM
- Add word pairs (bigrams) to the TF-IDF features
- Confusion matrix view for the most confused intents

## Author

**Karan Gojiya**, B.Tech CSE, IIIT Bhopal

[LinkedIn](https://www.linkedin.com/in/karan-gojiya) · [GitHub](https://github.com/KaranGojiya)
