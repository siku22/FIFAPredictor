from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def load_data(base_dir=BASE_DIR):
    players = pd.read_csv(base_dir / "fifa18_clean.csv", encoding="utf-8-sig")
    matches = pd.read_csv(base_dir / "international_matches.csv")
    rankings = pd.read_csv(base_dir / "fifa_mens_rank.csv")
    names = pd.read_csv(base_dir / "former_names.csv")

    return (
        clean_columns(players),
        clean_columns(matches),
        clean_columns(rankings),
        clean_columns(names),
    )


def build_team_features(players, rankings):
    team_features = (
        players.groupby("nationality")
        .agg(
            avg_overall=("overall", "mean"),
            avg_age=("age", "mean"),
            team_balance=("overall", "std"),
            num_players=("name", "count"),
        )
        .reset_index()
        .rename(columns={"nationality": "team"})
    )

    rankings = rankings.rename(
        columns={
            "total.points": "total_points",
            "previous.points": "previous_points",
            "diff.points": "diff_points",
        }
    ).copy()
    rankings["date"] = pd.to_datetime(rankings["date"].astype(str), format="%Y", errors="coerce")

    latest_rankings = rankings.sort_values("date").groupby("team", as_index=False).tail(1)

    team_features = team_features.merge(
        latest_rankings[["team", "rank", "total_points", "previous_points", "diff_points"]],
        on="team",
        how="left",
    )
    return team_features


def build_match_dataset(matches, names, team_features):
    name_map = dict(zip(names["former"], names["current"]))
    matches = matches.copy()
    matches["home_team"] = matches["home_team"].replace(name_map)
    matches["away_team"] = matches["away_team"].replace(name_map)
    matches["goal_diff"] = matches["home_goals"] - matches["away_goals"]
    matches["home_win"] = (matches["goal_diff"] > 0).astype(int)

    compiled = matches.merge(
        team_features.rename(columns=lambda col: f"home_{col}" if col != "team" else col),
        left_on="home_team",
        right_on="team",
        how="left",
    ).drop(columns="team")

    compiled = compiled.merge(
        team_features.rename(columns=lambda col: f"away_{col}" if col != "team" else col),
        left_on="away_team",
        right_on="team",
        how="left",
    ).drop(columns="team")

    compiled["overall_diff"] = compiled["home_avg_overall"] - compiled["away_avg_overall"]
    compiled["age_diff"] = compiled["home_avg_age"] - compiled["away_avg_age"]
    compiled["balance_diff"] = compiled["home_team_balance"] - compiled["away_team_balance"]
    compiled["player_pool_diff"] = compiled["home_num_players"] - compiled["away_num_players"]
    compiled["rank_advantage"] = compiled["away_rank"] - compiled["home_rank"]
    compiled["points_diff"] = compiled["home_total_points"] - compiled["away_total_points"]

    final_dataset = compiled[
        [
            "date",
            "tournament",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "goal_diff",
            "home_win",
            *FEATURES,
        ]
    ].dropna()

    final_dataset["date"] = pd.to_datetime(final_dataset["date"], errors="coerce")
    return final_dataset[final_dataset["date"].dt.year >= 2005].copy()


def train_model():
    players, matches, rankings, names = load_data()
    team_features = build_team_features(players, rankings)
    model_dataset = build_match_dataset(matches, names, team_features)

    X = model_dataset[FEATURES]
    y = model_dataset["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_dataset.to_csv(BASE_DIR / "compiled_fifa_match_dataset.csv", index=False)
    joblib.dump(model, BASE_DIR / "fifa_model.pkl")
    joblib.dump(team_features, BASE_DIR / "team_features.pkl")

    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "training_rows": int(len(model_dataset)),
        "available_prediction_teams": int(team_features[PREDICTION_COLUMNS].dropna().shape[0]),
    }
    joblib.dump(metrics, BASE_DIR / "model_metrics.pkl")

    return metrics


if __name__ == "__main__":
    metrics = train_model()
    print("Saved fifa_model.pkl, team_features.pkl, and compiled_fifa_match_dataset.csv")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Prediction-ready teams: {metrics['available_prediction_teams']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
