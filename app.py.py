import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# -----------------------------
# Try to use XGBoost, fall back to RandomForest if not installed
# -----------------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False


@st.cache_data
def load_data():
    df = pd.read_csv("in-vehicle-coupon-recommendation.csv")
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop("Y", axis=1)
    y = df["Y"]

    numeric_features = [
        "temperature",
        "has_children",
        "toCoupon_GEQ5min",
        "toCoupon_GEQ15min",
        "toCoupon_GEQ25min",
        "direction_same",
        "direction_opp",
    ]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X, y)

    # simple defaults: use most frequent value for each column
    defaults = {}
    for col in X.columns:
        if X[col].dtype == "O":
            defaults[col] = X[col].mode()[0]
        else:
            # numeric
            defaults[col] = float(X[col].median())

    return clf, defaults


def main():
    st.set_page_config(page_title="In-Vehicle Coupon Acceptance", layout="wide")
    st.title("In-Vehicle Coupon Acceptance – Demo App")
    st.write(
        "This app lets you enter a driving scenario and predicts whether the driver "
        "will accept an in-vehicle coupon (1 = accept, 0 = reject)."
    )

    clf, defaults = train_model()

    st.sidebar.header("Driving Scenario Inputs")

    destination = st.sidebar.selectbox(
        "Destination",
        ["No Urgent Place", "Home", "Work"],
        index=["No Urgent Place", "Home", "Work"].index(defaults["destination"]),
    )

    passanger = st.sidebar.selectbox(
        "Passenger",
        ["Alone", "Friend(s)", "Kid(s)", "Partner"],
        index=["Alone", "Friend(s)", "Kid(s)", "Partner"].index(defaults["passanger"]),
    )

    weather = st.sidebar.selectbox(
        "Weather",
        ["Sunny", "Rainy", "Snowy"],
        index=["Sunny", "Rainy", "Snowy"].index(defaults["weather"]),
    )

    time = st.sidebar.selectbox(
        "Time of day",
        ["7AM", "10AM", "2PM", "6PM", "10PM"],
        index=["7AM", "10AM", "2PM", "6PM", "10PM"].index(defaults["time"]),
    )

    temperature = st.sidebar.number_input(
        "Temperature (°F)", min_value=-10, max_value=120, value=int(defaults["temperature"])
    )

    coupon = st.sidebar.selectbox(
        "Coupon type",
        [
            "Restaurant(<20)",
            "Coffee House",
            "Carry out & Take away",
            "Bar",
            "Restaurant(20-50)",
        ],
        index=[
            "Restaurant(<20)",
            "Coffee House",
            "Carry out & Take away",
            "Bar",
            "Restaurant(20-50)",
        ].index(defaults["coupon"]),
    )

    expiration = st.sidebar.selectbox(
        "Coupon expiration",
        ["2h", "1d"],
        index=["2h", "1d"].index(defaults["expiration"]),
    )

    age = st.sidebar.selectbox(
        "Age",
        ["below21", "21", "26", "31", "36", "41", "46", "50plus"],
        index=["below21", "21", "26", "31", "36", "41", "46", "50plus"].index(
            defaults["age"]
        ),
    )

    income = st.sidebar.selectbox(
        "Income",
        [
            "Less than $12500",
            "$12500 - $24999",
            "$25000 - $37499",
            "$37500 - $49999",
            "$50000 - $62499",
            "$62500 - $74999",
            "$75000 - $87499",
            "$87500 - $99999",
            "$100000 or More",
        ],
        index=[
            "Less than $12500",
            "$12500 - $24999",
            "$25000 - $37499",
            "$37500 - $49999",
            "$50000 - $62499",
            "$62500 - $74999",
            "$75000 - $87499",
            "$87500 - $99999",
            "$100000 or More",
        ].index(defaults["income"]),
    )

    has_children = st.sidebar.checkbox(
        "Driver has children", value=bool(int(defaults["has_children"]))
    )

    # Frequency-type features
    freq_options = ["never", "less1", "1~3", "4~8", "gt8"]

    bar_freq = st.sidebar.selectbox(
        "Bar visit frequency",
        freq_options,
        index=freq_options.index(defaults["Bar"] if defaults["Bar"] in freq_options else "never"),
    )

    coffee_freq = st.sidebar.selectbox(
        "Coffee House visit frequency",
        freq_options,
        index=freq_options.index(
            defaults["CoffeeHouse"] if defaults["CoffeeHouse"] in freq_options else "never"
        ),
    )

    carry_freq = st.sidebar.selectbox(
        "Carry-out / Takeaway frequency",
        freq_options,
        index=freq_options.index(
            defaults["CarryAway"] if defaults["CarryAway"] in freq_options else "never"
        ),
    )

    rest20_freq = st.sidebar.selectbox(
        "Restaurant (<$20) frequency",
        freq_options,
        index=freq_options.index(
            defaults["RestaurantLessThan20"]
            if defaults["RestaurantLessThan20"] in freq_options
            else "never"
        ),
    )

    rest50_freq = st.sidebar.selectbox(
        "Restaurant ($20–$50) frequency",
        freq_options,
        index=freq_options.index(
            defaults["Restaurant20To50"]
            if defaults["Restaurant20To50"] in freq_options
            else "never"
        ),
    )

    # Distance & direction logic
    distance_label = st.sidebar.selectbox(
        "Driving time to coupon location",
        ["< 5 minutes", "5–15 minutes", "15–25 minutes"],
        index=0,
    )

    if distance_label == "< 5 minutes":
        to5, to15, to25 = 1, 0, 0
    elif distance_label == "5–15 minutes":
        to5, to15, to25 = 1, 1, 0
    else:
        to5, to15, to25 = 1, 1, 1

    same_direction = st.sidebar.checkbox(
        "Restaurant is in the same direction as current route", value=True
    )

    # Build input row using defaults + overrides from the sidebar
    input_data = defaults.copy()

    input_data.update(
        {
            "destination": destination,
            "passanger": passanger,  # note: column is spelled 'passanger' in the CSV
            "weather": weather,
            "time": time,
            "temperature": temperature,
            "coupon": coupon,
            "expiration": expiration,
            "age": age,
            "income": income,
            "has_children": int(has_children),
            "Bar": bar_freq,
            "CoffeeHouse": coffee_freq,
            "CarryAway": carry_freq,
            "RestaurantLessThan20": rest20_freq,
            "Restaurant20To50": rest50_freq,
            "toCoupon_GEQ5min": to5,
            "toCoupon_GEQ15min": to15,
            "toCoupon_GEQ25min": to25,
            "direction_same": 1 if same_direction else 0,
            "direction_opp": 0 if same_direction else 1,
        }
    )

    input_df = pd.DataFrame([input_data])

    st.subheader("Prediction")

    if st.button("Predict coupon acceptance"):
        proba = clf.predict_proba(input_df)[0, 1]
        pred = clf.predict(input_df)[0]

        st.metric(
            label="Probability of accepting coupon",
            value=f"{proba * 100:.1f} %",
        )

        if pred == 1:
            st.success("The model predicts: the driver **WILL ACCEPT** the coupon (Y = 1).")
        else:
            st.warning("The model predicts: the driver **WILL NOT ACCEPT** the coupon (Y = 0).")

        st.caption(
            "Note: this is a demo model trained on survey data. Predictions are estimates, "
            "not guarantees of real-world behavior."
        )

    with st.expander("Show raw input data sent to the model"):
        st.write(input_df)


if __name__ == "__main__":
    main()
