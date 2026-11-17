# ClearDesk — Customer Complaint Intelligence

ClearDesk is a small end-to-end machine-learning web app that helps a customer-support team make sense of incoming complaints.

A support agent enters a complaint and the app suggests:
- the complaint category
- a priority level (High, Medium or Low)
- the customer's sentiment
- the team that should review it

It also includes a session dashboard with charts and a CSV export.

## Tech stack

- Python
- pandas
- scikit-learn (TF-IDF + Logistic Regression)
- Streamlit
- Plotly

## Run locally

Use Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Install packages
pip install -r requirements.txt

# 3. Train the models
python train_model.py

# 4. Start the app
streamlit run app.py
```

The app can also train the models automatically the first time it starts.

## Project structure

```text
complaint-intelligence-system/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── complaints.csv
└── models/
    └── (created when models are trained)
```

## How it works

1. `TfidfVectorizer` turns complaint text into numeric features.
2. A Logistic Regression pipeline predicts the complaint category.
3. A second Logistic Regression pipeline predicts the suggested priority.
4. A small word-list sentiment check labels text as positive, neutral or negative.
5. A routing map recommends a support department.

## Dataset note

The included CSV is a small, hand-written demonstration dataset. It is intended to make the project runnable and explainable, not to represent real customer traffic. The model is not a production-grade classifier and its predictions should be reviewed by a person.

For a stronger version, collect a larger consented dataset, remove personal information, label it consistently, use a held-out test set, and report per-class precision, recall and F1-score.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud: https://share.streamlit.io/
3. Choose **Create app** and select your repository.
4. Set the main file path to `app.py`.
5. Deploy.

No API key is required.

## Suggested GitHub description

A lightweight NLP and machine-learning app that classifies customer complaints, suggests priority, checks sentiment and recommends a support department. Built with Python, scikit-learn and Streamlit.

## Limitations and responsible use

- Predictions depend on a small demonstration dataset and may be wrong.
- Sentiment analysis is deliberately simple and may miss sarcasm, context and mixed feelings.
- Do not enter real customer personal, payment, medical or account-security information into this demo.
- Do not use the priority prediction as the only basis for urgent decisions.
