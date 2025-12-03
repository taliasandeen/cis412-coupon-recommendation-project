import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.neural_network import MLPClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Coupon Models – XGBoost vs Neural Net", layout="wide")
st.title("Coupon Acceptance Models: XGBoost vs Neural Network")

st.write(
    "This app uses the **in-vehicle-coupon-recommendation.csv** dataset to explore "
    "coupon acceptance behavior and compare an XGBoost model to a Neural Network."
)

if not XGBOOST_AVAILABLE:
    st.warning(
        "⚠️ `xgboost` is not installed. The XGBoost model and comparison sections "
        "will be limited until you install it with `pip install xgboost`."
    )

# -----------------------------
# 1. DATA LOADING & PREP
# -----------------------------

@st.cache_data
def load_and_prepare_data(csv_path: str):
    # Load raw data
    df = pd.read_csv(csv_path)

    # Convert all object columns to categorical
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    # Optional: map numeric temperature to labels, then treat as categorical
    if "temperature" in df.columns and np.issubdtype(df["temperature"].dtype, np.number):
        df["temperature"] = df["temperature"].map(
            {30: "Cold", 55: "Cool", 80: "Warm"}
        )
        df["temperature"] = df["temperature"].astype("category")

    # Make sure target exists
    if "Y" not in df.columns:
        raise ValueError("Dataset does not include column 'Y' (target variable).")

    # Drop columns too high-cardinality or not useful
    drop_cols = ["occupation"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Feature / target split
    X = df.drop(columns=["Y"] + drop_cols)
    y = df["Y"]

    # Auto-detect categorical vs numeric
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    num_cols = X.select_dtypes(exclude="category").columns.tolist()

    # One-hot encode ALL categorical columns
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_ohe = ohe.fit_transform(X[cat_cols])
    ohe_cols = ohe.get_feature_names_out(cat_cols)
    X_ohe = pd.DataFrame(X_ohe, columns=ohe_cols, index=X.index)

    # Final numeric feature matrix
    X_final = pd.concat([X[num_cols], X_ohe], axis=1)

    # Ensure final dataset is numeric only
    if X_final.select_dtypes(include="object").shape[1] > 0:
        raise ValueError("Non-numeric columns remain after encoding.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature names for later plots
    feature_names = X_final.columns.tolist()

    # Scale for NN
    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train.values)
    X_test_nn = scaler.transform(X_test.values)

    return df, X_train, X_test, y_train, y_test, feature_names, X_train_nn, X_test_nn


# Sidebar input for CSV path
st.sidebar.header("Data Settings")
default_path = "in-vehicle-coupon-recommendation.csv"
csv_path = st.sidebar.text_input(
    "Path to CSV file",
    value=default_path,
    help="Update this if your CSV is in a different folder.",
)

# Try loading data
data_loaded = True
try:
    (
        coupon,
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        feature_names,
        X_train_nn,
        X_test_nn,
    ) = load_and_prepare_data(csv_path)
except Exception as e:
    st.error(f"Error loading data from `{csv_path}`: {e}")
    data_loaded = False

if not data_loaded:
    st.stop()  # Don't run the rest of the app if data failed to load

st.success("✅ Data loaded successfully.")
st.write(f"Training samples: {X_train_final.shape[0]}, Features: {X_train_final.shape[1]}")

# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------

def plot_overall_acceptance(df):
    y_counts = df["Y"].value_counts().sort_index()
    labels = ["Not Accepted (0)", "Accepted (1)"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        y_counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#6495ED", "#ADD8E6"],
    )
    ax.set_title("Overall Coupon Acceptance")
    return fig

def plot_acceptance_by_coupon_type(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("coupon")["Y"].mean().sort_values().plot(
        kind="bar", color="steelblue", ax=ax
    )
    ax.set_title("Acceptance Rate by Coupon Type")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_acceptance_by_time(df):
    time_order = ["7AM", "10AM", "2PM", "6PM", "10PM"]
    acc_time = df.groupby("time")["Y"].mean().reindex(time_order)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(x=acc_time.index, y=acc_time.values, color="steelblue", ax=ax)
    ax.set_title("Acceptance Rate by Time of Day")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    plt.tight_layout()
    return fig

def plot_acceptance_across_coupon_and_time(df):
    acc = df.groupby(["coupon", "time"])["Y"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=acc,
        x="time",
        y="Y",
        hue="coupon",
        marker="o",
        palette="Blues",
        ax=ax,
    )
    ax.set_title("Acceptance Rate Across Coupon Types and Times")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time of Day")
    ax.legend(title="Coupon Type", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig

def plot_acceptance_by_passenger(df):
    # Handle both 'passenger' and the common misspelling 'passanger'
    if "passenger" in df.columns:
        col = "passenger"
    elif "passanger" in df.columns:
        col = "passanger"
    else:
        # Fallback: show a message instead of crashing
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No passenger/passanger column in data.",
                ha="center", va="center")
        ax.axis("off")
        return fig

    acc_pass = df.groupby(col)["Y"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=acc_pass.values,
        y=acc_pass.index,
        hue=acc_pass.index,
        palette="Blues",
        legend=False,
        ax=ax,
    )
    ax.set_title("Acceptance Rate by Passenger Type")
    ax.set_xlabel("Acceptance Rate")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Passenger Type")
    plt.tight_layout()
    return fig


def plot_acceptance_by_income(df):
    income_order = [
        "Less than $12500",
        "$12500 - $24999",
        "$25000 - $37499",
        "$37500 - $49999",
        "$50000 - $62499",
        "$62500 - $74999",
        "$75000 - $87499",
        "$87500 - $99999",
        "$100000 or More",
    ]
    if "income" in df.columns:
        df = df.copy()
        df["income"] = pd.Categorical(df["income"], categories=income_order, ordered=True)
        acc_income = df.groupby("income")["Y"].mean().sort_values()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(
            x=acc_income.values,
            y=acc_income.index,
            hue=acc_income.index,
            palette="Blues",
            legend=False,
            ax=ax,
        )
        ax.set_title("Acceptance Rate by Income Level")
        ax.set_xlabel("Acceptance Rate")
        ax.set_ylabel("Income Level")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        return fig
    return None

def compute_metrics(y_true, y_pred, y_proba, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    report_dict = classification_report(
        y_true, y_pred, target_names=["Not Accepted", "Accepted"], output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    return {
        "name": model_name,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_curve": precision,
        "recall_curve": recall,
        "report_df": report_df,
    }

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Accepted", "Accepted"],
        yticklabels=["Not Accepted", "Accepted"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def plot_pr_curve(precision, recall, auc_value, label):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"{label} (AUPRC = {auc_value:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig

def plot_feature_importance(importances, feature_names, title, top_n=10):
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = imp_series.head(top_n)[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig

def get_nn_feature_importance(nn_model, feature_names):
    """Approximate NN feature importance from absolute input-layer weights."""
    # nn_model.coefs_[0]: shape (n_features, n_hidden)
    coefs = nn_model.coefs_[0]
    importance = np.mean(np.abs(coefs), axis=1)
    return importance

# -----------------------------
# 3. TRAIN MODELS
# -----------------------------

@st.cache_resource
def train_xgb_model(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        return None

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        max_depth=5,
        learning_rate=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        n_estimators=200,
    )
    # Use .values to avoid feature name issues
    model.fit(X_train.values, y_train)
    return model

@st.cache_resource
def train_nn_model(X_train_scaled, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.005,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    model.fit(X_train_scaled, y_train)
    return model

xgb_model = train_xgb_model(X_train_final, y_train) if XGBOOST_AVAILABLE else None
nn_model = train_nn_model(X_train_nn, y_train)

# XGBoost predictions & metrics
xgb_results = None
if xgb_model is not None:
    y_test_pred_xgb = xgb_model.predict(X_test_final.values)
    y_test_proba_xgb = xgb_model.predict_proba(X_test_final.values)[:, 1]
    xgb_results = compute_metrics(
        y_test, y_test_pred_xgb, y_test_proba_xgb, model_name="XGBoost"
    )

# Neural Network predictions & metrics
y_test_pred_nn = nn_model.predict(X_test_nn)
y_test_proba_nn = nn_model.predict_proba(X_test_nn)[:, 1]
nn_results = compute_metrics(
    y_test, y_test_pred_nn, y_test_proba_nn, model_name="Neural Network"
)

# NN feature importance (fast, no permutation importance)
nn_importances = get_nn_feature_importance(nn_model, feature_names)

# -----------------------------
# 4. LAYOUT: TABS
# -----------------------------

tab_eda, tab_xgb, tab_nn, tab_compare = st.tabs(
    ["EDA", "XGBoost", "Neural Network", "Model Comparison"]
)

# ----- EDA TAB -----
with tab_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall Coupon Acceptance")
        fig = plot_overall_acceptance(coupon)
        st.pyplot(fig)

    with col2:
        st.subheader("Acceptance Rate by Coupon Type")
        fig = plot_acceptance_by_coupon_type(coupon)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Acceptance Rate by Time of Day")
        fig = plot_acceptance_by_time(coupon)
        st.pyplot(fig)

    with col4:
        st.subheader("Acceptance Rate by Passenger Type")
        fig = plot_acceptance_by_passenger(coupon)
        st.pyplot(fig)

    st.subheader("Acceptance Rate Across Coupon Types and Times")
    fig = plot_acceptance_across_coupon_and_time(coupon)
    st.pyplot(fig)

    st.subheader("Acceptance Rate by Income Level")
    fig_income = plot_acceptance_by_income(coupon)
    if fig_income is not None:
        st.pyplot(fig_income)
    else:
        st.info("Income column not found in data.")

# ----- XGBOOST TAB -----
with tab_xgb:
    st.header("XGBoost Model Results")

    if xgb_model is None:
        st.warning(
            "XGBoost is not available. Install `xgboost` to enable this section."
        )
    else:
        st.subheader("Classification Report")
        st.dataframe(xgb_results["report_df"].style.format("{:.3f}"))

        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(
            y_test, y_test_pred_xgb, "XGBoost Confusion Matrix"
        )
        st.pyplot(fig)

        st.subheader("ROC Curve")
        fig = plot_roc_curve(
            y_test, y_test_proba_xgb, "XGBoost"
        )
        st.pyplot(fig)

        st.subheader("Precision–Recall Curve")
        fig = plot_pr_curve(
            xgb_results["precision_curve"],
            xgb_results["recall_curve"],
            xgb_results["pr_auc"],
            "XGBoost",
        )
        st.pyplot(fig)

        st.subheader("Feature Importance (Top 10)")
        xgb_importances = xgb_model.feature_importances_
        fig = plot_feature_importance(
            xgb_importances, feature_names, "XGBoost Feature Importance (Top 10)"
        )
        st.pyplot(fig)

# ----- NEURAL NETWORK TAB -----
with tab_nn:
    st.header("Neural Network Model Results")

    st.subheader("Classification Report")
    st.dataframe(nn_results["report_df"].style.format("{:.3f}"))

    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(
        y_test, y_test_pred_nn, "Neural Network Confusion Matrix"
    )
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fig = plot_roc_curve(y_test, y_test_proba_nn, "Neural Network")
    st.pyplot(fig)

    st.subheader("Precision–Recall Curve")
    fig = plot_pr_curve(
        nn_results["precision_curve"],
        nn_results["recall_curve"],
        nn_results["pr_auc"],
        "Neural Network",
    )
    st.pyplot(fig)

    st.subheader("Feature Importance (Top 10 – Approx from NN Weights)")
    fig = plot_feature_importance(
        nn_importances,
        feature_names,
        "Neural Network Feature Importance (Top 10)",
    )
    st.pyplot(fig)

# ----- COMPARISON TAB -----
with tab_compare:
    st.header("Model Comparison: XGBoost vs Neural Network")

    rows = []
    rows.append(
        {
            "Model": "Neural Network",
            "Accuracy": nn_results["accuracy"],
            "ROC AUC": nn_results["roc_auc"],
            "PR AUC": nn_results["pr_auc"],
        }
    )
    if xgb_results is not None:
        rows.append(
            {
                "Model": "XGBoost",
                "Accuracy": xgb_results["accuracy"],
                "ROC AUC": xgb_results["roc_auc"],
                "PR AUC": xgb_results["pr_auc"],
            }
        )

    compare_df = pd.DataFrame(rows).set_index("Model")
    st.subheader("Summary Metrics")
    st.dataframe(compare_df.style.format("{:.3f}"))

    if xgb_results is not None:
        st.subheader("Precision–Recall Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            nn_results["recall_curve"],
            nn_results["precision_curve"],
            lw=2,
            label=f"Neural Network (AUPRC = {nn_results['pr_auc']:.3f})",
        )
        ax.plot(
            xgb_results["recall_curve"],
            xgb_results["precision_curve"],
            lw=2,
            label=f"XGBoost (AUPRC = {xgb_results['pr_auc']:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve: XGBoost vs Neural Network")
        ax.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Install `xgboost` to see a direct comparison between models.")
