"""
Telemarketing_streamlit.py
Improved dashboard for the Bank-Marketing campaign data
"""

# ─────────────────────────── imports & style ────────────────────────────
import streamlit as st
import plotly.express as px
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)

sns.set_theme(style="whitegrid")          # nicer default look
plt.rcParams.update({"axes.titlesize": 15,
                    "axes.labelsize": 12})

st.set_page_config(page_title="Bank Marketing Campaign Dashboard",
                   layout="wide",
                   page_icon="📈")

# ─────────────────────────────── file paths ─────────────────────────────
DATA_PATH  = r"bank_marketing_data.csv"
IMAGE_PATH = r"bank.jpeg"

# ───────────────────────────── data helpers ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=3)

def _duration_to_min(x: str) -> float:
    if pd.isna(x):
        return np.nan
    val = float(str(x).strip().split()[0])
    return val / 60 if "sec" in str(x).lower() else val

@st.cache_data(show_spinner=False)
def clean_data(df0: pd.DataFrame) -> pd.DataFrame:
    df = df0.copy()

    # split jobedu
    df[["job", "education"]] = df["jobedu"].str.split(",", expand=True)
    df.drop(columns=["jobedu"], inplace=True)

    # duration
    df["duration_min"] = df["duration"].apply(_duration_to_min)
    df.drop(columns=["duration"], inplace=True)

    # drop customerid, age_band
    df.drop(columns=[c for c in ["customerid", "age_band"] if c in df.columns],
            inplace=True)

    # impute age + month
    df["age"]   = df["age"].fillna(df["age"].median()).astype(int)
    df["month"] = df["month"].fillna(df["month"].mode()[0])

    # engineered features
    df["response_flag"] = df["response"].map({"yes": 1, "no": 0})
    df["was_contacted_before"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)
    df["pdays"] = df["pdays"].fillna(999)

    # yes/no binaries
    for col in ["targeted", "default", "housing", "loan"]:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # one-hot encode
    multi = ["marital", "job", "education", "contact", "month", "poutcome"]
    df = pd.get_dummies(df, columns=multi, drop_first=True)

    return df

# ───────────────────────── baseline rule ────────────────────────────────
def baseline_rule(X: pd.DataFrame) -> np.ndarray:
    req = ["education_tertiary", "marital_single",
           "was_contacted_before", "loan"]
    if not set(req).issubset(X.columns):
        return np.zeros(len(X), dtype=int)
    mask = (
        (X["education_tertiary"] == 1) &
        (X["marital_single"] == 1) &
        (X["was_contacted_before"] == 1) &
        (X["loan"] == 0)
    )
    return np.where(mask, 1, 0)

# ─────────────────────────── model trainer ──────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    df = df.dropna(subset=["response_flag"])
    X = df.drop(columns=["response_flag", "response"])
    y = df["response_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    # baseline predictions
    X_train["baseline_pred"] = baseline_rule(X_train)
    X_test ["baseline_pred"] = baseline_rule(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(class_weight="balanced",
                                 max_iter=8000,
                                 random_state=42)
    log_reg.fit(X_train.drop(columns=["baseline_pred"]), y_train)

    # Random-Forest (tiny grid)
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid = {"n_estimators":[300], "max_depth":[None, 10], "min_samples_leaf":[1, 4]}
    gs = GridSearchCV(rf, grid, cv=3, n_jobs=-1, scoring="roc_auc")
    gs.fit(X_train.drop(columns=["baseline_pred"]), y_train)

    return {
        "X_test_full": X_test,
        "X_test":      X_test.drop(columns=["baseline_pred"]),
        "y_test":      y_test,
        "log_reg":     log_reg,
        "rf":          gs.best_estimator_,
    }

# ─────────────────────── metric helper ──────────────────────────────────
def mstats(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    return {"acc": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "cm":  confusion_matrix(y_test, y_pred)}

# ═════════════════════════ UI PAGES ════════════════════════════════════
def page_home():
    # header first
    st.markdown("## Bank Marketing Campaign Dashboard")
    st.image(Image.open(IMAGE_PATH), use_column_width=True)
    st.success(
        "Use the sidebar to explore: **Overview → Data Prep → EDA → Models → Predict**"
    )

def page_overview():
    st.header("📋 Overview")
    st.write(
        """
        This app walks through the full data-science pipeline used to
        analyse a Portuguese bank’s direct-marketing campaign:

        1. **Data Prep** – see every cleaning & feature-engineering step  
        2. **EDA** – interactive visuals to uncover patterns  
        3. **Models** – baseline rule, Logistic Regression & Random Forest  
        4. **Predict** – enter a prospect’s details and get a success probability  

        Navigate with the radio buttons in the sidebar.
        """
    )

def page_prep(raw, clean):
    st.header("⚙️ Data preparation")
    st.subheader("Before vs. after cleaning")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Raw snapshot")
        st.dataframe(raw.head())
    with col2:
        st.caption("Cleaned snapshot")
        st.dataframe(clean.head())

    st.subheader("Transformation steps")
    steps = [
        "Split `jobedu` ➜ `job`, `education`",
        "Convert `duration` to minutes (`duration_min`)",
        "Drop `customerid`, `age_band`, original `duration`",
        "Impute **age** (median) & **month** (mode)",
        "Map yes/no binaries to 1/0",
        "Create `response_flag` (target)",
        "Engineer `was_contacted_before`",
        "Fill `pdays` NaNs with 999",
        "One-hot encode: marital, job, education, contact, month, poutcome",
    ]
    st.markdown("• " + "\n• ".join(steps))

def page_eda(df):
    st.header("🔎 Exploratory data analysis")

    # 1. Target balance ─────────────────────────────────────────
    cnts = df["response_flag"].value_counts().rename({0: "No", 1: "Yes"})
    fig = px.bar(cnts, x=cnts.index, y=cnts.values,
                 labels={"x": "Response", "y": "Count"},
                 title="Target balance",
                 color=cnts.index, color_discrete_sequence=["#d62728", "#2ca02c"])
    st.plotly_chart(fig, use_container_width=True)

    # 2. Call-duration KDE ─────────────────────────────────────
    fig = px.histogram(df, x="duration_min", color="response_flag",
                       nbins=50, barmode="overlay", histnorm="probability density",
                       labels={"duration_min": "Duration (min)",
                               "response_flag": "Response"},
                       title="Call duration distribution",
                       color_discrete_map={0: "#d62728", 1: "#2ca02c"})
    fig.update_traces(opacity=0.55)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Success rate by previous campaign outcome ─────────────
    pout_cols = [c for c in df.columns if c.startswith("poutcome_")]
    rates = {c.replace("poutcome_", ""):
             df[df[c] == 1]["response_flag"].mean() for c in pout_cols}
    sr = pd.Series(rates).sort_values(ascending=False)
    fig = px.bar(sr, x=sr.index, y=sr.values,
                 labels={"x": "Previous outcome", "y": "Success rate"},
                 title="Success rate vs. previous campaign outcome",
                 color=sr.values, color_continuous_scale="viridis")
    st.plotly_chart(fig, use_container_width=True)


def page_models(art):
    st.header("🤖 Model performance")

    # baseline numbers
    y_base = baseline_rule(art["X_test_full"])
    acc_b = accuracy_score(art["y_test"], y_base)
    st.write(f"**Baseline-rule accuracy** : `{acc_b:.3f}`")

    # loop through LR & RF
    for name, mdl in {"Logistic Regression": art["log_reg"],
                      "Random Forest":      art["rf"]}.items():
        stats = mstats(mdl, art["X_test"], art["y_test"])
        st.subheader(name)
        st.write(f"Accuracy **{stats['acc']:.3f}**  |  AUC **{stats['auc']:.3f}**")

        # interactive confusion matrix
        cm_df = pd.DataFrame(stats["cm"],
                             index=["Actual 0", "Actual 1"],
                             columns=["Pred 0", "Pred 1"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                        title=f"{name} – Confusion matrix")
        st.plotly_chart(fig, use_container_width=False)

    # ROC curves
    roc_fig = px.line(title="ROC curves")
    roc_fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          yaxis=dict(scaleanchor="x", scaleratio=1))
    for lbl, mdl in [("Logistic Regression", art["log_reg"]),
                     ("Random Forest",       art["rf"])]:
        fpr, tpr, _ = roc_curve(art["y_test"],
                                mdl.predict_proba(art["X_test"])[:, 1])
        roc_fig.add_scatter(x=fpr, y=tpr, mode="lines", name=lbl)
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dash"), name="Random guess")
    st.plotly_chart(roc_fig, use_container_width=True)


def page_predict(art):
    st.header("🎯 Predict a new prospect")

    X_cols = art["X_test"].columns.tolist()
    row = pd.DataFrame(columns=X_cols)

    with st.form("prediction"):
        # basic numeric + binary fields
        row.loc[0,"age"]          = st.slider("Age",18,95,35)
        row.loc[0,"salary"]       = st.number_input("Salary",0,step=1000,value=60000)
        row.loc[0,"balance"]      = st.number_input("Balance",0,step=100,value=1000)
        row.loc[0,"duration_min"] = st.number_input("Last‐call duration (min)",
                                                    value=2.0)
        row.loc[0,"campaign"]     = st.slider("Campaign contacts so far",1,63,2)
        pdays = st.number_input("Days since last contact (-1 = never)",value=-1)
        row.loc[0,"pdays"] = 999 if pdays==-1 else pdays
        row.loc[0,"was_contacted_before"] = 0 if pdays==-1 else 1

        row.loc[0,"housing"] = 1 if st.radio("Has housing loan?",["No","Yes"])=="Yes" else 0
        row.loc[0,"loan"]    = 1 if st.radio("Has personal loan?",["No","Yes"])=="Yes" else 0
        row.loc[0,"default"] = 1 if st.radio("Has credit in default?",["No","Yes"])=="Yes" else 0

        # one-hot placeholders
        for col in X_cols:
            if row[col].isna().any():
                row[col] = 0

        st.form_submit_button("Predict")

    if not row[X_cols].isnull().values.any():
        prob_rf = art["rf"].predict_proba(row)[0,1]
        prob_lr = art["log_reg"].predict_proba(row)[0,1]
        st.success(f"Random Forest probability ➡️ **{prob_rf:.2%}**")
        st.info   (f"Logistic Regression probability ➡️ **{prob_lr:.2%}**")

# ══════════════════════════  main  ══════════════════════════════════════
def main():
    raw   = load_data(DATA_PATH)
    clean = clean_data(raw)
    art   = train_models(clean)

    pages = {"🏠 Home":      page_home,
             "📋 Overview":  page_overview,
             "⚙️ Data Prep": lambda: page_prep(raw, clean),
             "🔍 EDA":       lambda: page_eda(clean),
             "🤖 Models":    lambda: page_models(art),
             "🎯 Predict":   lambda: page_predict(art)}

    with st.sidebar:
        st.markdown("## Navigation")
        choice = st.radio("", list(pages.keys()))

    pages[choice]()

if __name__ == "__main__":
    main()
