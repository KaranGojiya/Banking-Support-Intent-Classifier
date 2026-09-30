import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Banking Support Intent Classifier",
    page_icon="🏦",
    layout="wide",
)

# =============================================================
# Model performance (measured on the 3,076-query Banking77 test split)
# =============================================================
MODEL_METRICS = {
    "Test accuracy": "93.6%",
    "Macro F1-score": "93.6%",
    "Top-3 accuracy": "98.4%",
    "Intents": "77",
}

# =============================================================
# Load model
# =============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("banking_intent_classifier.pkl")


try:
    model = load_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that banking_intent_classifier.pkl "
        "is in the app folder and that scikit-learn 1.6.1 is installed."
    )
    st.exception(exc)
    st.stop()

# =============================================================
# Support team for every one of the 77 intents
# =============================================================
CARD = "Card Support Team"
SECURITY = "Card Security Team"
ATM = "ATM Support Team"
PAYMENT = "Payment Support Team"
TRANSFER = "Transfer Support Team"
TOPUP = "Top-Up Support Team"
ACCOUNT = "Account Support Team"
FRAUD = "Fraud & Transaction Support Team"
EXCHANGE = "Exchange & Currency Team"
IDENTITY = "Identity Verification Team"
REFUND = "Refund Support Team"

department_map = {
    "Refund_not_showing_up": REFUND,
    "activate_my_card": CARD,
    "age_limit": ACCOUNT,
    "apple_pay_or_google_pay": PAYMENT,
    "atm_support": ATM,
    "automatic_top_up": TOPUP,
    "balance_not_updated_after_bank_transfer": TRANSFER,
    "balance_not_updated_after_cheque_or_cash_deposit": ACCOUNT,
    "beneficiary_not_allowed": TRANSFER,
    "cancel_transfer": TRANSFER,
    "card_about_to_expire": CARD,
    "card_acceptance": CARD,
    "card_arrival": CARD,
    "card_delivery_estimate": CARD,
    "card_linking": CARD,
    "card_not_working": CARD,
    "card_payment_fee_charged": PAYMENT,
    "card_payment_not_recognised": FRAUD,
    "card_payment_wrong_exchange_rate": EXCHANGE,
    "card_swallowed": ATM,
    "cash_withdrawal_charge": ATM,
    "cash_withdrawal_not_recognised": FRAUD,
    "change_pin": SECURITY,
    "compromised_card": SECURITY,
    "contactless_not_working": PAYMENT,
    "country_support": ACCOUNT,
    "declined_card_payment": PAYMENT,
    "declined_cash_withdrawal": ATM,
    "declined_transfer": TRANSFER,
    "direct_debit_payment_not_recognised": FRAUD,
    "disposable_card_limits": CARD,
    "edit_personal_details": ACCOUNT,
    "exchange_charge": EXCHANGE,
    "exchange_rate": EXCHANGE,
    "exchange_via_app": EXCHANGE,
    "extra_charge_on_statement": PAYMENT,
    "failed_transfer": TRANSFER,
    "fiat_currency_support": EXCHANGE,
    "get_disposable_virtual_card": CARD,
    "get_physical_card": CARD,
    "getting_spare_card": CARD,
    "getting_virtual_card": CARD,
    "lost_or_stolen_card": SECURITY,
    "lost_or_stolen_phone": FRAUD,
    "order_physical_card": CARD,
    "passcode_forgotten": ACCOUNT,
    "pending_card_payment": PAYMENT,
    "pending_cash_withdrawal": ATM,
    "pending_top_up": TOPUP,
    "pending_transfer": TRANSFER,
    "pin_blocked": SECURITY,
    "receiving_money": TRANSFER,
    "request_refund": REFUND,
    "reverted_card_payment?": PAYMENT,
    "supported_cards_and_currencies": CARD,
    "terminate_account": ACCOUNT,
    "top_up_by_bank_transfer_charge": TOPUP,
    "top_up_by_card_charge": TOPUP,
    "top_up_by_cash_or_cheque": TOPUP,
    "top_up_failed": TOPUP,
    "top_up_limits": TOPUP,
    "top_up_reverted": TOPUP,
    "topping_up_by_card": TOPUP,
    "transaction_charged_twice": FRAUD,
    "transfer_fee_charged": TRANSFER,
    "transfer_into_account": TRANSFER,
    "transfer_not_received_by_recipient": TRANSFER,
    "transfer_timing": TRANSFER,
    "unable_to_verify_identity": IDENTITY,
    "verify_my_identity": IDENTITY,
    "verify_source_of_funds": IDENTITY,
    "verify_top_up": TOPUP,
    "virtual_card_not_working": CARD,
    "visa_or_mastercard": CARD,
    "why_verify_identity": IDENTITY,
    "wrong_amount_of_cash_received": ATM,
    "wrong_exchange_rate_for_cash_withdrawal": EXCHANGE,
}

EXAMPLE_QUERIES = [
    "Why was I charged twice for one payment?",
    "The exchange rate on my payment looks wrong",
    "My card has not arrived yet",
    "Someone stole my card",
]

STOP_WORDS = {
    "i", "am", "is", "are", "the", "my", "me", "to", "for", "a", "an",
    "and", "or", "of", "in", "on", "it", "this", "that", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "have",
    "has", "had", "not", "still", "yet",
}


# =============================================================
# Helper functions
# =============================================================
def format_intent(intent):
    return intent.replace("?", "").replace("_", " ").strip().capitalize()


def extract_keywords(query):
    vocabulary = model.named_steps["tfidf"].vocabulary_
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    keywords = [w for w in words if w not in STOP_WORDS and w in vocabulary]
    return list(dict.fromkeys(keywords))  # remove repeats, keep order


# Bands and accuracies were measured on the Banking77 test split:
#   confidence 50% or more -> 98.6% of predictions correct (77% of queries)
#   confidence 30% to 50%  -> 84.5% correct (16% of queries)
#   confidence below 30%   -> 59.8% correct (7% of queries)
BAND_ACCURACY = {"High": "98.6%", "Medium": "84.5%", "Low": "59.8%"}


def confidence_band(pct):
    if pct >= 50:
        return "High"
    if pct >= 30:
        return "Medium"
    return "Low"


def analyze(query):
    probabilities = model.predict_proba([query])[0]
    class_names = [str(c) for c in model.classes_]

    order = np.argsort(probabilities)[::-1]
    top_index = order[0]
    second_index = order[1]

    intent = class_names[top_index]
    confidence = float(probabilities[top_index] * 100)

    others = [
        {"intent": class_names[i], "confidence": float(probabilities[i] * 100)}
        for i in order[1:6]
    ]

    return {
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "margin": confidence - float(probabilities[second_index] * 100),
        "team": department_map.get(intent, "General Banking Support"),
        "keywords": extract_keywords(query),
        "others": others,
    }


def use_example(text):
    st.session_state["query"] = text


# =============================================================
# Session state
# =============================================================
st.session_state.setdefault("query", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])

# =============================================================
# Header
# =============================================================
st.title("🏦 Banking Support Intent Classifier")
st.caption(
    "Paste a customer message and the model predicts what the customer needs, "
    "then suggests which support team should handle it. Built with TF-IDF and "
    "Multinomial Naive Bayes on the Banking77 dataset."
)

st.divider()

# =============================================================
# Input
# =============================================================
st.subheader("Customer query")

user_query = st.text_area(
    "Enter the customer message",
    key="query",
    placeholder="Example: Someone stole my card...",
    height=110,
)

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, text in zip(example_cols, EXAMPLE_QUERIES):
    col.button(text, on_click=use_example, args=(text,), key=f"ex_{text}")

analyze_btn = st.button("🔍 Analyze query", type="primary")

if analyze_btn:
    cleaned = " ".join(user_query.split())
    if cleaned == "":
        st.warning("Please enter a customer query.")
    elif len(cleaned) < 5:
        st.warning("That message is too short. Add a few more words so the model has something to work with.")
    else:
        result = analyze(cleaned)
        st.session_state["result"] = result
        st.session_state["history"].insert(
            0,
            {
                "Query": cleaned,
                "Predicted intent": format_intent(result["intent"]),
                "Confidence (%)": round(result["confidence"], 1),
                "Suggested team": result["team"],
            },
        )
        st.session_state["history"] = st.session_state["history"][:10]

# =============================================================
# Results
# =============================================================
result = st.session_state["result"]

if result is None:
    st.info("The prediction will appear here after you analyze a query.")
else:
    st.divider()
    left_col, right_col = st.columns([1.3, 1], gap="large")

    with left_col:
        st.subheader("Prediction")

        st.success(f"This query is most likely about **{format_intent(result['intent'])}**.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{result['confidence']:.1f}%")
        m2.metric("Confidence level", confidence_band(result["confidence"]))
        m3.metric(
            "Lead over next intent",
            f"{result['margin']:.1f} pts",
            help="Confidence of the top intent minus the confidence of the runner-up. "
                 "A small gap means the model is torn between two intents.",
        )

        band = confidence_band(result["confidence"])
        st.caption(
            f"On the test set, {band.lower()}-confidence predictions were correct "
            f"{BAND_ACCURACY[band]} of the time."
        )

        if band == "Low":
            st.warning(
                "Low confidence. Ask the customer for more detail or route this "
                "to a human agent."
            )

        st.markdown("**Suggested team**")
        st.info(result["team"])

        st.markdown("**Keywords detected**")
        if result["keywords"]:
            st.markdown(" ".join(f"`{word}`" for word in result["keywords"]))
        else:
            st.caption("No strong keywords found in this message.")

    with right_col:
        st.subheader("Other possible intents")
        for item in result["others"]:
            st.markdown(f"**{format_intent(item['intent'])}**")
            st.progress(min(max(item["confidence"] / 100, 0.0), 1.0))
            st.caption(f"{item['confidence']:.2f}% confidence")

# =============================================================
# History
# =============================================================
if st.session_state["history"]:
    st.divider()
    with st.expander(f"Recent queries ({len(st.session_state['history'])})"):
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

# =============================================================
# Model performance
# =============================================================
st.divider()
st.subheader("Model performance")
st.caption("Measured on the 3,076 test queries of the Banking77 dataset.")

perf_cols = st.columns(len(MODEL_METRICS))
for col, (name, value) in zip(perf_cols, MODEL_METRICS.items()):
    col.metric(name, value)

st.caption(
    "Top-3 accuracy means the correct intent appears among the model's three "
    "highest-ranked guesses."
)

# =============================================================
# Footer
# =============================================================
st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | "
    "[LinkedIn](https://www.linkedin.com/in/karan-gojiya)"
)
