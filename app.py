import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import train_models

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"
CATEGORY_MODEL = MODEL_DIR / "category_model.joblib"
PRIORITY_MODEL = MODEL_DIR / "priority_model.joblib"
DATA_FILE = APP_DIR / "data" / "complaints.csv"

st.set_page_config(
    page_title="ClearDesk | Complaint Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A small, explainable word list keeps sentiment analysis lightweight.
POSITIVE_WORDS = {
    "good", "great", "excellent", "helpful", "happy", "love", "quick",
    "resolved", "thank", "thanks", "perfect", "friendly", "appreciate",
}
NEGATIVE_WORDS = {
    "bad", "angry", "upset", "poor", "terrible", "hate", "broken",
    "delay", "delayed", "missing", "wrong", "failed", "failure", "refund",
    "charged", "crash", "crashes", "unacceptable", "disappointed",
    "frustrated", "issue", "problem", "defective", "stolen", "hacked",
}

DEPARTMENT_MAP = {
    "Billing & Payments": "Finance and Billing",
    "Delivery & Shipping": "Logistics",
    "Technical Support": "Technical Support",
    "Product Quality": "Quality and Returns",
    "Account & Security": "Account Security",
    "General Enquiry": "Customer Service",
}

def get_sentiment(text):
    """Return a simple sentiment label and score based on word matches."""
    words = re.findall(r"[a-z']+", text.lower())
    positive_count = sum(word in POSITIVE_WORDS for word in words)
    negative_count = sum(word in NEGATIVE_WORDS for word in words)
    score = positive_count - negative_count

    if score > 0:
        return "Positive", score
    if score < 0:
        return "Negative", score
    return "Neutral", score


def load_models():
    """Train the models on first launch, then reuse the saved files."""
    if not CATEGORY_MODEL.exists() or not PRIORITY_MODEL.exists():
        train_models()
    return joblib.load(CATEGORY_MODEL), joblib.load(PRIORITY_MODEL)


@st.cache_resource
def cached_models():
    return load_models()


def analyse_complaint(complaint, category_model, priority_model):
    category = category_model.predict([complaint])[0]
    priority = priority_model.predict([complaint])[0]
    sentiment, sentiment_score = get_sentiment(complaint)
    return {
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "department": DEPARTMENT_MAP.get(category, "Customer Service"),
    }


# ------------------------------ Sidebar ------------------------------------
with st.sidebar:
    st.title("🧭 ClearDesk")
    st.caption("Customer Complaint Intelligence")
    st.markdown("---")
    page = st.radio("Workspace", ["Analyse a complaint", "Team dashboard", "About"])
    st.markdown("---")
    st.caption("Built with Python, NLP and scikit-learn")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(110deg, #102a43, #245b75);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; color: #e5eef5;}
    div[data-testid="stMetric"] {
        background: #f5f8fb; padding: 14px 16px;
        border-radius: 12px; border: 1px solid #e3eaf0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

category_model, priority_model = cached_models()

if page == "Analyse a complaint":
    st.markdown(
        """
        <div class="hero">
            <h1>Understand every customer complaint.</h1>
            <p>Turn a message into a clear category, urgency level and next step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Complaint details")
    complaint = st.text_area(
        "Paste or type the customer's message",
        placeholder="Example: I was charged twice and my refund has not arrived...",
        height=150,
        max_chars=2000,
    )
    analyse_clicked = st.button("Analyse complaint", type="primary", use_container_width=True)

    if analyse_clicked:
        if len(complaint.strip()) < 8:
            st.warning("Please enter a little more detail so the model has something useful to analyse.")
        else:
            result = analyse_complaint(complaint, category_model, priority_model)
            st.session_state["latest_result"] = result
            st.session_state["latest_complaint"] = complaint

    if "latest_result" in st.session_state:
        result = st.session_state["latest_result"]
        st.markdown("---")
        st.subheader("Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Complaint category", result["category"])
        col2.metric("Suggested priority", result["priority"])
        col3.metric("Customer sentiment", result["sentiment"])

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Recommended next step")
            st.info(
                f"Route this complaint to **{result['department']}**. "
                "A team member should review the message and confirm the suggested classification."
            )
            if result["priority"] == "High":
                st.error("Suggested action: review this complaint as soon as possible.")
            elif result["priority"] == "Medium":
                st.warning("Suggested action: add this complaint to the team's active queue.")
            else:
                st.success("Suggested action: handle through the normal support workflow.")
        with right:
            st.markdown("#### What the model picked up")
            st.write(f"**Category:** {result['category']}")
            st.write(f"**Priority:** {result['priority']}")
            st.write(f"**Sentiment:** {result['sentiment']}")
            st.caption(
                "These are model-assisted suggestions, not a final decision. "
                "A support agent should verify sensitive or urgent cases."
            )

        if st.button("Add this example to the session dashboard"):
            entry = {
                "Complaint": st.session_state["latest_complaint"],
                "Category": result["category"],
                "Priority": result["priority"],
                "Sentiment": result["sentiment"],
                "Department": result["department"],
            }
            history = st.session_state.get("complaint_history", [])
            history.append(entry)
            st.session_state["complaint_history"] = history
            st.success("Added to this session's dashboard.")

elif page == "Team dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Team overview</h1>
            <p>A quick view of the complaints analysed in this browser session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    history = st.session_state.get("complaint_history", [])
    if not history:
        st.info("No complaints have been added yet. Analyse a complaint and select 'Add this example to the session dashboard'.")
    else:
        data = pd.DataFrame(history)
        high_count = int((data["Priority"] == "High").sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Complaints analysed", len(data))
        col2.metric("High priority", high_count)
        col3.metric("Negative sentiment", int((data["Sentiment"] == "Negative").sum()))

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.pie(data, names="Category", title="Complaints by category", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            fig = px.bar(
                data["Priority"].value_counts().rename_axis("Priority").reset_index(name="Count"),
                x="Priority", y="Count", title="Priority distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Recent complaints")
        st.dataframe(data, use_container_width=True, hide_index=True)
        csv_data = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download session report (CSV)", csv_data, "complaint_report.csv", "text/csv")

elif page == "About":
    st.markdown(
        """
        <div class="hero">
            <h1>About ClearDesk</h1>
            <p>A practical machine-learning project for customer support teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### What it does
        ClearDesk uses text classification to suggest a complaint category and priority.
        It also applies a small, transparent word-based sentiment check and recommends
        the team that should review the case.

        ### How the machine-learning part works
        - Complaint text is cleaned and converted into TF-IDF features.
        - A scikit-learn classifier predicts the complaint category.
        - A second classifier predicts the suggested priority.
        - Sentiment is estimated using a small word list so the logic is easy to inspect.

        ### Important limitation
        This project uses a small, hand-curated demonstration dataset. Its predictions are
        suitable for a portfolio demonstration, not for making real customer, financial,
        security or emergency decisions. A production system needs a larger, representative,
        reviewed dataset, proper evaluation, privacy controls and human oversight.
        """
    )
