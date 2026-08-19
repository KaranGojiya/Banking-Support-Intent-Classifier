# 🏦 Banking Support Intent Classifier

A machine learning web application that predicts the intent behind customer banking support queries using **TF-IDF Vectorization** and **Multinomial Naive Bayes**.

The app allows a user to enter a banking-related query, predicts the most likely customer intent, shows the confidence score, highlights important detected keywords, and displays other possible intents with confidence levels.

---

## 🚀 Project Overview

Customer support teams receive thousands of queries every day. Manually understanding and routing each query to the correct support department can be time-consuming.

This project solves that problem by automatically classifying customer banking queries into different intent categories such as card arrival, lost card, cash withdrawal issue, contactless payment issue, PIN blocked, and more.

---

## ✨ Features

- 💬 Chatbot-style customer query interface
- 🎯 Predicts the best matching banking intent
- 📊 Shows confidence score for the prediction
- 🔑 Displays important keywords detected from the query
- 📈 Shows other possible intents using progress bars
- 🏢 Suggests the relevant support team
- ⚡ Fast prediction using a trained ML pipeline
- 🌐 Interactive Streamlit web app

---

## 🧠 Machine Learning Model

The model is built using:

- **TF-IDF Vectorizer** for converting text into numerical features
- **Multinomial Naive Bayes** for multiclass intent classification

Final tuned model parameters:

```python
TfidfVectorizer(
    max_df=0.9,
    min_df=2,
    ngram_range=(1, 1),
    sublinear_tf=True
)

MultinomialNB(alpha=0.1)
