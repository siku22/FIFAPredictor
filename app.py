from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "overall_diff",
    "age_diff",
    "balance_diff",
    "player_pool_diff",
    "rank_advantage",
    "points_diff",
]
PREDICTION_COLUMNS = [
    "avg_overall",
    "avg_age",
    "team_balance",
    "num_players",
    "rank",
    "total_points",
]


@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "fifa_model.pkl"
    features_path = BASE_DIR / "team_features.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python3 train_model.py` first."
        )

    return joblib.load(model_path), joblib.load(features_path)


def prediction_ready_teams(team_features):
    ready = team_features.dropna(subset=PREDICTION_COLUMNS)
    return sorted(ready["team"].dropna().unique())


def predict_match(team_a_name, team_b_name, team_features, model):
    tf = team_features.copy()
    tf["team_clean"] = tf["team"].str.lower().str.strip()

    team_a = tf[tf["team_clean"] == team_a_name.lower().strip()].iloc[0]
    team_b = tf[tf["team_clean"] == team_b_name.lower().strip()].iloc[0]

    input_data = pd.DataFrame(
        {
            "overall_diff": [team_a["avg_overall"] - team_b["avg_overall"]],
            "age_diff": [team_a["avg_age"] - team_b["avg_age"]],
            "balance_diff": [team_a["team_balance"] - team_b["team_balance"]],
            "player_pool_diff": [team_a["num_players"] - team_b["num_players"]],
            "rank_advantage": [team_b["rank"] - team_a["rank"]],
            "points_diff": [team_a["total_points"] - team_b["total_points"]],
        }
    )[FEATURES]

    prob_a = model.predict_proba(input_data)[0][1]
    prob_b = 1 - prob_a

    return {
        "winner": team_a_name if prob_a > prob_b else team_b_name,
        "team_a_probability": prob_a,
        "team_b_probability": prob_b,
    }


st.set_page_config(page_title="2026 FIFA Predictor", page_icon="soccer", layout="wide")
st.title("2026 FIFA World Cup Predictor")
st.caption("Predict match outcomes using FIFA rankings, player ratings, and team composition.")
st.divider()

try:
    model, team_features = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

teams = prediction_ready_teams(team_features)

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    default_b = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
    team_b = st.selectbox("Team B", teams, index=default_b)

if st.button("Predict", use_container_width=True):
    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        result = predict_match(team_a, team_b, team_features, model)
        winner = result["winner"]
        prob_a = result["team_a_probability"]
        prob_b = result["team_b_probability"]

        st.subheader(f"{winner} is more likely to win")
        metric_a, metric_b = st.columns(2)
        with metric_a:
            st.metric(team_a, f"{prob_a:.2%}")
        with metric_b:
            st.metric(team_b, f"{prob_b:.2%}")

        st.progress(float(prob_a), text=f"{team_a} win probability")
        st.success(f"Prediction favors {winner}")

st.divider()
st.caption("Model: Logistic Regression | Data: international matches from 2005 onward")
