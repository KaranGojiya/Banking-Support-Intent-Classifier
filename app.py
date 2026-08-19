import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

st.set_page_config(
    page_title="Banking Support Intent Classifier",
    page_icon="🏦",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("banking_intent_classifier.pkl")

model = load_model()

# Support Team Mapping
department_map = {
    "card_arrival": "Card Support Team",
    "lost_or_stolen_card": "Card Security Team",
    "card_not_working": "Card Support Team",
    "contactless_not_working": "Payment Support Team",
    "cash_withdrawal_charge": "ATM Support Team",
    "pending_cash_withdrawal": "ATM Support Team",
    "wrong_amount_of_cash_received": "ATM Support Team",
    "cash_withdrawal_card": "ATM Support Team",
    "balance_not_updated_after_cheque_or_cash_deposit": "Account Support Team",
    "direct_debit_payment_not_recognised": "Fraud & Transaction Support Team",
    "pin_blocked": "Card Security Team",
    "cash_withdrawal": "ATM Support Team",
    "beneficiary_not_verified": "Transfer Support Team",
    "bank_card_missing": "Card Support Team",
    "cash_withdrawal_pending": "ATM Support Team",
    "cash_withdrawal_cancelled": "ATM Support Team",
    "card_acceptance": "Card Support Team",
    "card_swallowed": "ATM Support Team",
    "cash_withdrawal_exchange_rate": "ATM Support Team",
    "cash_withdrawal_reverted": "ATM Support Team",
    "cash_withdrawal_wrong_exchange_rate": "ATM Support Team",
    "cash_withdrawal_cardless": "ATM Support Team"
}

# Helper Functions
def format_intent(intent):
    return intent.replace("_", " ").title()

def extract_keywords(query, model):
    tfidf = model.named_steps["tfidf"]
    vocabulary = tfidf.vocabulary_

    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())

    stop_words = {
        "i", "am", "is", "are", "the", "my", "me", "to", "for", "a", "an",
        "and", "or", "of", "in", "on", "it", "this", "that", "was", "were",
        "do", "does", "did", "can", "could", "would", "should", "have",
        "has", "had", "not", "still", "yet"
    }

    keywords = []

    for word in words:
        if word not in stop_words and word in vocabulary:
            keywords.append(word)

    return list(dict.fromkeys(keywords))

# Page Header
st.title("🏦 Banking Support Intent Classifier")
st.caption("AI-powered customer query analysis using TF-IDF and Multinomial Naive Bayes.")

st.divider()

# Session State
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "top_predictions" not in st.session_state:
    st.session_state.top_predictions = []

if "keywords" not in st.session_state:
    st.session_state.keywords = []

# Layout
left_col, center_col, right_col = st.columns([1.2, 2, 1.2])

# Center Column - Chatbot UI
with center_col:
    st.subheader("💬 Customer Query Assistant")

    user_query = st.text_area(
        "Enter customer query",
        placeholder="Example: Someone stole my card...",
        height=130
    )

    analyze_btn = st.button("🔍 Analyze Query", use_container_width=True)

    if analyze_btn:
        if user_query.strip() == "":
            st.warning("Please enter a customer query.")
        else:
            prediction = model.predict([user_query])[0]
            probabilities = model.predict_proba([user_query])[0]
            confidence = probabilities.max() * 100

            class_names = model.classes_
            top_indices = np.argsort(probabilities)[-6:][::-1]

            top_predictions = []
            for idx in top_indices:
                top_predictions.append({
                    "intent": class_names[idx],
                    "confidence": probabilities[idx] * 100
                })

            keywords = extract_keywords(user_query, model)

            st.session_state.prediction_done = True
            st.session_state.user_query = user_query
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.top_predictions = top_predictions
            st.session_state.keywords = keywords
    if st.session_state.prediction_done:
        st.markdown("#### Customer Query")
        st.info(st.session_state.user_query)

        st.markdown("#### AI Assistant Response")
        st.success(
            f"This query is most likely related to **{format_intent(st.session_state.prediction)}**."
        )

# Left Column - Main Prediction
with left_col:
    st.subheader("🎯 Main Prediction")

    if st.session_state.prediction_done:
        st.markdown("#### Predicted Intent")
        st.success(format_intent(st.session_state.prediction))

        st.metric(
            "Confidence Rate",
            f"{st.session_state.confidence:.2f}%"
        )

        department = department_map.get(
            st.session_state.prediction,
            "General Banking Support"
        )

        st.markdown("#### Suggested Team")
        st.info(department)

        st.markdown("#### 🔑 Keywords Detected")

        if len(st.session_state.keywords) > 0:
            keyword_text = " ".join(
                [f"`{word}`" for word in st.session_state.keywords]
            )
            st.markdown(keyword_text)
        else:
            st.warning("No strong keywords detected.")
    else:
        st.info("Prediction result will appear here after analyzing a query.")

# Right Column - Other Choices
with right_col:
    st.subheader("Other Possible Intents")

    if st.session_state.prediction_done:
        other_predictions = [
            item for item in st.session_state.top_predictions
            if item["intent"] != st.session_state.prediction
        ]

        for item in other_predictions[:5]:
            intent_name = format_intent(item["intent"])
            confidence_value = item["confidence"]

            st.markdown(f"**{intent_name}**")
            st.progress(confidence_value / 100)
            st.caption(f"{confidence_value:.2f}% confidence")
            st.write("")
    else:
        st.info("Alternative intents will appear here.")

st.divider()

st.caption(
    "This system predicts banking support intents using a trained TF-IDF and Multinomial Naive Bayes pipeline."
)

st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | [LinkedIn](https://linkedin.com/in/karan-gojiya)"
)