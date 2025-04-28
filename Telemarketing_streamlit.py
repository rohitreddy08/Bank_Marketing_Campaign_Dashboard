"""
Telemarketing_streamlit.py
Dashboard for the Bank-Marketing campaign — now loading *pre-trained* models.
"""

# ───────────────────────── imports & style ─────────────────────────
import streamlit as st, plotly.express as px, joblib, pickle
from PIL import Image
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 12})

st.set_page_config(page_title="Bank Marketing Campaign Dashboard",
                   layout="wide", page_icon="📈")

# ─────────────────────────── file paths ───────────────────────────
DATA_PATH  = "bank_marketing_data.csv"
IMAGE_PATH = "bank.jpeg"

# ───────────────────────── data helpers ───────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=3)

def _duration_to_min(x):
    if pd.isna(x): return np.nan
    v=float(str(x).strip().split()[0]); return v/60 if "sec" in str(x).lower() else v

@st.cache_data(show_spinner=False)
def clean_data(df0):
    df=df0.copy()
    df[["job","education"]]=df["jobedu"].str.split(",",expand=True); df.drop(columns="jobedu",inplace=True)
    df["duration_min"]=df["duration"].apply(_duration_to_min); df.drop(columns="duration",inplace=True)
    df.drop(columns=[c for c in ("customerid","age_band") if c in df.columns],inplace=True)
    df["age"]=df["age"].fillna(df["age"].median()).astype(int)
    df["month"]=df["month"].fillna(df["month"].mode()[0])
    df["response_flag"]=df["response"].map({"yes":1,"no":0})
    df["was_contacted_before"]=df["pdays"].apply(lambda x:0 if x==-1 else 1)
    df["pdays"]=df["pdays"].fillna(999)
    for col in ("targeted","default","housing","loan"):
        df[col]=df[col].map({"yes":1,"no":0})
    df=pd.get_dummies(df,columns=["marital","job","education","contact","month","poutcome"],drop_first=True)
    return df

# ───────────────────────── baseline rule ──────────────────────────
def baseline_rule(X:pd.DataFrame):
    req=["education_tertiary","marital_single","was_contacted_before","loan"]
    if not set(req).issubset(X.columns): return np.zeros(len(X),dtype=int)
    m=(X["education_tertiary"]&X["marital_single"]&X["was_contacted_before"]&(X["loan"]==0))
    return m.astype(int).values

# ─────────────────────── model loader (fast) ─────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    log_reg = joblib.load("log_reg.pkl")
    rf      = joblib.load("rf.pkl")
    with open("feature_cols.pkl","rb") as f:
        cols = pickle.load(f)
    return {"log_reg":log_reg,"rf":rf,"cols":cols}

# ───────────────────── metric convenience fn ────────────────────
def mstats(model,X,y):
    pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    return {"acc":accuracy_score(y,pred),
            "auc":roc_auc_score(y,prob),
            "cm":confusion_matrix(y,pred)}

# ═════════════════════ UI PAGES ══════════════════════
def page_home():
    st.markdown("## Bank Marketing Campaign Dashboard")
    st.image(Image.open(IMAGE_PATH),use_container_width=True)
    st.success("Use the sidebar to explore: **Overview → Data Prep → EDA → Models → Predict**")

def page_overview():
    st.header("📋 Overview")
    st.write("""
    This app walks through the full data-science pipeline used to analyse a Portuguese
    bank’s direct-marketing campaign:

    1. **Data Prep** – all cleaning & feature engineering  
    2. **EDA** – interactive visuals  
    3. **Models** – baseline rule, Logistic Regression & Random Forest  
    4. **Predict** – score a new prospect
    """)

def page_prep(raw,clean):
    st.header("⚙️ Data preparation")
    c1,c2=st.columns(2)
    with c1: st.caption("Raw snapshot");   st.dataframe(raw.head())
    with c2: st.caption("Cleaned snapshot"); st.dataframe(clean.head())
    st.subheader("Transformation steps")
    st.markdown("• "+"\n• ".join([
        "Split jobedu ➜ job & education",
        "duration → minutes (duration_min)",
        "Drop customerid / age_band",
        "Impute age (median) & month (mode)",
        "Map yes/no binaries to 1/0",
        "Create response_flag (target)",
        "Engineer was_contacted_before",
        "Fill pdays NaNs with 999",
        "One-hot encode categorical vars"
    ]))

def page_eda(df):
    st.header("🔎 Exploratory data analysis")
    # bar target
    cnt=df["response_flag"].value_counts().rename({0:"No",1:"Yes"})
    st.plotly_chart(px.bar(cnt,x=cnt.index,y=cnt.values,title="Target balance",
                           labels={"x":"Response","y":"Count"},
                           color=cnt.index,color_discrete_sequence=["#d62728","#2ca02c"]),
                    use_container_width=True)
    # duration hist
    h=px.histogram(df,x="duration_min",color="response_flag",nbins=50,barmode="overlay",
                   histnorm="probability density",opacity=0.55,
                   labels={"duration_min":"Duration (min)","response_flag":"Response"},
                   title="Call duration distribution",
                   color_discrete_map={0:"#d62728",1:"#2ca02c"})
    st.plotly_chart(h,use_container_width=True)
    # success by poutcome
    pout=[c for c in df.columns if c.startswith("poutcome_")]
    rates={c[9:]:df.loc[df[c]==1,"response_flag"].mean() for c in pout}
    sr=pd.Series(rates).sort_values(ascending=False)
    st.plotly_chart(px.bar(sr,x=sr.index,y=sr.values,title="Success rate vs previous outcome",
                           labels={"x":"Previous outcome","y":"Success rate"},
                           color=sr.values,color_continuous_scale="viridis"),
                    use_container_width=True)

def page_models(models,clean):
    st.header("🤖 Model performance")
    df_eval=clean.dropna(subset=["response_flag"])
    y=df_eval["response_flag"].values
    X=df_eval[models["cols"]]
    acc_b=accuracy_score(y,baseline_rule(df_eval))
    st.write(f"**Baseline-rule accuracy** : {acc_b:.3f}")

    for label,mdl in {"Logistic Regression":models["log_reg"],
                      "Random Forest":models["rf"]}.items():
        s=mstats(mdl,X,y)
        st.subheader(label)
        st.write(f"Accuracy **{s['acc']:.3f}** | AUC **{s['auc']:.3f}**")
        cm=pd.DataFrame(s["cm"],index=["Actual 0","Actual 1"],
                        columns=["Pred 0","Pred 1"])
        st.plotly_chart(px.imshow(cm,text_auto=True,color_continuous_scale="Blues",
                                  title=f"{label} – Confusion matrix"),
                        use_container_width=False)

    roc=px.line(title="ROC curves")
    roc.update_layout(xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      yaxis=dict(scaleanchor="x",scaleratio=1))
    for lbl,mdl in [("Logistic Regression",models["log_reg"]),
                    ("Random Forest",models["rf"])]:
        fpr,tpr,_=roc_curve(y,mdl.predict_proba(X)[:,1])
        roc.add_scatter(x=fpr,y=tpr,mode="lines",name=lbl)
    roc.add_scatter(x=[0,1],y=[0,1],mode="lines",
                    line=dict(dash="dash"),name="Random guess")
    st.plotly_chart(roc,use_container_width=True)

def page_predict(models):
    st.header("🎯 Predict a new prospect")
    cols=models["cols"]
    row=pd.DataFrame(columns=cols)
    with st.form("prediction"):
        row.loc[0,"age"]=st.slider("Age",18,95,35)
        row.loc[0,"salary"]=st.number_input("Salary",0,step=1000,value=60000)
        row.loc[0,"balance"]=st.number_input("Balance",0,step=100,value=1000)
        row.loc[0,"duration_min"]=st.number_input("Last-call duration (min)",2.0)
        row.loc[0,"campaign"]=st.slider("Campaign contacts so far",1,63,2)
        pdays=st.number_input("Days since last contact (-1 = never)",-1)
        row.loc[0,"pdays"]=999 if pdays==-1 else pdays
        row.loc[0,"was_contacted_before"]=0 if pdays==-1 else 1
        row.loc[0,"housing"]=1 if st.radio("Housing loan?",["No","Yes"])=="Yes" else 0
        row.loc[0,"loan"]=1 if st.radio("Personal loan?",["No","Yes"])=="Yes" else 0
        row.loc[0,"default"]=1 if st.radio("Credit in default?",["No","Yes"])=="Yes" else 0
        for c in cols:
            if row[c].isna().any(): row[c]=0
        st.form_submit_button("Predict")
    if not row[cols].isnull().values.any():
        st.success(f"Random Forest probability ➡ **{models['rf'].predict_proba(row[cols])[0,1]:.2%}**")
        st.info   (f"Logistic Regression probability ➡ **{models['log_reg'].predict_proba(row[cols])[0,1]:.2%}**")

# ────────────────────────────── main ─────────────────────────────
def main():
    raw   = load_data(DATA_PATH)
    clean = clean_data(raw)
    models= load_models()

    pages={
        "🏠 Home":      page_home,
        "📋 Overview":  page_overview,
        "⚙️ Data Prep": lambda: page_prep(raw,clean),
        "🔍 EDA":       lambda: page_eda(clean),
        "🤖 Models":    lambda: page_models(models,clean),
        "🎯 Predict":   lambda: page_predict(models),
    }

    with st.sidebar:
        st.markdown("## Navigation")
        choice=st.radio("",list(pages.keys()))
    pages[choice]()

if __name__=="__main__":
    main()
