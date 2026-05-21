from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

FEATURES = [
    "overall_diff",
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "goalkeeper_diff",
    "age_diff",
    "balance_diff",
    "player_pool_diff",
    "elite_player_diff",
    "rank_advantage",
    "points_diff",
    "strength_diff",
    "spine_diff",
    "rank_points_blend",
    "attack_vs_defense_gap",
    "abs_strength_gap",
]

PREDICTION_COLUMNS = [
    "avg_overall",
    "attack_strength",
    "midfield_strength",
    "defense_strength",
    "goalkeeper_strength",
    "avg_age",
    "team_balance",
    "num_players",
    "elite_players",
    "rank",
    "total_points",
]

BASE_MATCH_FEATURES = [
    "overall_diff",
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "goalkeeper_diff",
    "age_diff",
    "balance_diff",
    "player_pool_diff",
    "elite_player_diff",
    "rank_advantage",
    "points_diff",
]

TEAM_NAME_ALIASES = {
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Turkey": "Türkiye",
    "USA": "United States",
}

QUALIFIED_2026_TEAMS = [
    "Canada",
    "Mexico",
    "United States",
    "Australia",
    "Iraq",
    "Iran",
    "Japan",
    "Jordan",
    "South Korea",
    "Qatar",
    "Saudi Arabia",
    "Uzbekistan",
    "Algeria",
    "Cape Verde",
    "DR Congo",
    "Ivory Coast",
    "Egypt",
    "Ghana",
    "Morocco",
    "Senegal",
    "South Africa",
    "Tunisia",
    "Curaçao",
    "Haiti",
    "Panama",
    "Argentina",
    "Brazil",
    "Colombia",
    "Ecuador",
    "Paraguay",
    "Uruguay",
    "New Zealand",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Croatia",
    "Czechia",
    "England",
    "France",
    "Germany",
    "Netherlands",
    "Norway",
    "Portugal",
    "Scotland",
    "Spain",
    "Sweden",
    "Switzerland",
    "Türkiye",
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


def load_data(raw_data_dir=RAW_DATA_DIR):
    players = pd.read_csv(raw_data_dir / "fifa18_clean.csv", encoding="utf-8-sig")
    matches = pd.read_csv(raw_data_dir / "international_matches.csv")
    rankings = pd.read_csv(raw_data_dir / "fifa_mens_rank.csv")
    names = pd.read_csv(raw_data_dir / "former_names.csv")
    return clean_columns(players), clean_columns(matches), clean_columns(rankings), clean_columns(names)


def position_mean(group, mask_column):
    values = group.loc[group[mask_column], "overall"]
    return values.mean() if len(values) else group["overall"].mean()


def build_team_features(players, rankings):
    players = players.copy()
    players["nationality"] = players["nationality"].replace(TEAM_NAME_ALIASES)
    players["preferred_positions"] = players["preferred_positions"].fillna("")
    players["is_forward"] = players["preferred_positions"].str.contains("ST|CF|LW|RW|LF|RF", regex=True)
    players["is_midfielder"] = players["preferred_positions"].str.contains("CM|CAM|CDM|LM|RM", regex=True)
    players["is_defender"] = players["preferred_positions"].str.contains("CB|LB|RB|LWB|RWB", regex=True)
    players["is_goalkeeper"] = players["preferred_positions"].str.contains("GK", regex=True)

    team_features = (
        players.groupby("nationality")
        .agg(
            avg_overall=("overall", "mean"),
            avg_age=("age", "mean"),
            team_balance=("overall", "std"),
            num_players=("name", "count"),
            elite_players=("overall", lambda values: int((values >= 80).sum())),
        )
        .reset_index()
        .rename(columns={"nationality": "team"})
    )
    strengths = (
        players.groupby("nationality")
        .apply(
            lambda group: pd.Series(
                {
                    "attack_strength": position_mean(group, "is_forward"),
                    "midfield_strength": position_mean(group, "is_midfielder"),
                    "defense_strength": position_mean(group, "is_defender"),
                    "goalkeeper_strength": position_mean(group, "is_goalkeeper"),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"nationality": "team"})
    )
    team_features = team_features.merge(strengths, on="team", how="left")

    rankings = rankings.rename(
        columns={
            "total.points": "total_points",
            "previous.points": "previous_points",
            "diff.points": "diff_points",
        }
    ).copy()
    rankings["team"] = rankings["team"].replace(TEAM_NAME_ALIASES)
    rankings["date"] = pd.to_datetime(rankings["date"].astype(str), format="%Y", errors="coerce")
    latest_rankings = rankings.sort_values("date").groupby("team", as_index=False).tail(1)

    team_features = team_features.merge(
        latest_rankings[["team", "rank", "total_points", "previous_points", "diff_points"]],
        on="team",
        how="left",
    )

    median_features = team_features[PREDICTION_COLUMNS[:-2]].median(numeric_only=True)
    for team in QUALIFIED_2026_TEAMS:
        if team in set(team_features["team"]):
            continue
        rank_row = latest_rankings[latest_rankings["team"] == team]
        fallback = {
            **median_features.to_dict(),
            "team": team,
            "rank": pd.NA,
            "total_points": pd.NA,
            "previous_points": pd.NA,
            "diff_points": pd.NA,
        }
        if not rank_row.empty:
            fallback.update(rank_row.iloc[0][["rank", "total_points", "previous_points", "diff_points"]].to_dict())
        team_features = pd.concat([team_features, pd.DataFrame([fallback])], ignore_index=True)

    return team_features


def build_match_dataset(matches, names, team_features):
    name_map = dict(zip(names["former"], names["current"]))
    matches = matches.copy()
    matches["home_team"] = matches["home_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["away_team"] = matches["away_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
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
    compiled["attack_diff"] = compiled["home_attack_strength"] - compiled["away_attack_strength"]
    compiled["midfield_diff"] = compiled["home_midfield_strength"] - compiled["away_midfield_strength"]
    compiled["defense_diff"] = compiled["home_defense_strength"] - compiled["away_defense_strength"]
    compiled["goalkeeper_diff"] = compiled["home_goalkeeper_strength"] - compiled["away_goalkeeper_strength"]
    compiled["age_diff"] = compiled["home_avg_age"] - compiled["away_avg_age"]
    compiled["balance_diff"] = compiled["home_team_balance"] - compiled["away_team_balance"]
    compiled["player_pool_diff"] = compiled["home_num_players"] - compiled["away_num_players"]
    compiled["elite_player_diff"] = compiled["home_elite_players"] - compiled["away_elite_players"]
    compiled["rank_advantage"] = compiled["away_rank"] - compiled["home_rank"]
    compiled["points_diff"] = compiled["home_total_points"] - compiled["away_total_points"]
    compiled = add_engineered_match_features(compiled)

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


def add_engineered_match_features(df):
    df = df.copy()
    df["strength_diff"] = df[
        ["overall_diff", "attack_diff", "midfield_diff", "defense_diff", "goalkeeper_diff"]
    ].mean(axis=1)
    df["spine_diff"] = df[["midfield_diff", "defense_diff", "goalkeeper_diff"]].mean(axis=1)
    df["rank_points_blend"] = (df["rank_advantage"] * 0.03) + (df["points_diff"] * 0.001)
    df["attack_vs_defense_gap"] = df["attack_diff"] - df["defense_diff"]
    df["abs_strength_gap"] = df["strength_diff"].abs()
    return df


def classification_metrics(y_test, y_pred, y_prob=None):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def evaluate_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Polynomial Logistic Regression": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(C=0.2, max_iter=4000),
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Naive Bayes": GaussianNB(),
        "MLP Classifier": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1200, random_state=42),
        ),
    }

    trained = {}
    results = {}
    for name, candidate in models.items():
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)
        y_prob = candidate.predict_proba(X_test)[:, 1] if hasattr(candidate, "predict_proba") else None
        trained[name] = candidate
        results[name] = classification_metrics(y_test, y_pred, y_prob)

    best_name = max(results, key=lambda name: results[name]["accuracy"])
    return trained, results, best_name


def evaluate_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Dummy Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10000)),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "SVR": make_pipeline(StandardScaler(), SVR()),
        "MLP Regressor": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1200, random_state=42),
        ),
    }

    results = {}
    for name, candidate in models.items():
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "r2": r2_score(y_test, y_pred),
        }
    return results


def build_tournament_success_dataset(world_cup_matches, names, team_features):
    name_map = dict(zip(names["former"], names["current"]))
    matches = clean_columns(world_cup_matches)
    matches["home_team"] = matches["home_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["away_team"] = matches["away_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["stage"] = matches["stage"].str.lower()

    group_teams = (
        matches[matches["stage"].eq("group stage")]
        .melt(id_vars=["year"], value_vars=["home_team", "away_team"], value_name="team")[["year", "team"]]
        .drop_duplicates()
    )
    knockout_teams = (
        matches[~matches["stage"].eq("group stage")]
        .melt(id_vars=["year"], value_vars=["home_team", "away_team"], value_name="team")[["year", "team"]]
        .drop_duplicates()
    )
    dataset = group_teams.merge(
        knockout_teams.assign(advanced_past_group=1),
        on=["year", "team"],
        how="left",
    )
    dataset["advanced_past_group"] = dataset["advanced_past_group"].fillna(0).astype(int)
    dataset = dataset.merge(team_features, on="team", how="left").dropna(subset=PREDICTION_COLUMNS)
    return dataset[dataset["year"] >= 2006].copy()


def evaluate_tournament_success_models(success_dataset):
    if len(success_dataset) < 30 or success_dataset["advanced_past_group"].nunique() < 2:
        return {}
    X = success_dataset[PREDICTION_COLUMNS]
    y = success_dataset["advanced_past_group"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "MLP Classifier": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=1200, random_state=42),
        ),
    }

    results = {}
    for name, candidate in models.items():
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)
        y_prob = candidate.predict_proba(X_test)[:, 1] if hasattr(candidate, "predict_proba") else None
        results[name] = classification_metrics(y_test, y_pred, y_prob)
    return results


def extract_feature_importance(model, feature_names):
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        return []
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .to_dict("records")
    )


def train_model():
    players, matches, rankings, names = load_data()
    team_features = build_team_features(players, rankings)
    model_dataset = build_match_dataset(matches, names, team_features)

    X = model_dataset[FEATURES]
    trained_models, classification_results, best_model_name = evaluate_classification_models(
        X, model_dataset["home_win"]
    )
    regression_results = evaluate_regression_models(X, model_dataset["goal_diff"])

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    world_cup_matches = pd.read_csv(RAW_DATA_DIR / "world_cup_matches.csv")
    tournament_success_dataset = build_tournament_success_dataset(world_cup_matches, names, team_features)
    tournament_success_results = evaluate_tournament_success_models(tournament_success_dataset)

    model = trained_models[best_model_name]
    model_dataset.to_csv(PROCESSED_DATA_DIR / "compiled_fifa_match_dataset.csv", index=False)
    tournament_success_dataset.to_csv(PROCESSED_DATA_DIR / "tournament_success_dataset.csv", index=False)
    joblib.dump(model, MODELS_DIR / "fifa_model.pkl")
    joblib.dump(team_features, MODELS_DIR / "team_features.pkl")

    feature_importance_model = trained_models.get("Random Forest", model)

    metrics = {
        "best_match_model": best_model_name,
        "accuracy": classification_results[best_model_name]["accuracy"],
        "classification_report": classification_results[best_model_name]["classification_report"],
        "confusion_matrix": classification_results[best_model_name]["confusion_matrix"],
        "classification_models": classification_results,
        "regression_models": regression_results,
        "tournament_success_models": tournament_success_results,
        "feature_importance_model": "Random Forest",
        "feature_importance": extract_feature_importance(feature_importance_model, FEATURES),
        "training_rows": int(len(model_dataset)),
        "tournament_success_rows": int(len(tournament_success_dataset)),
        "available_prediction_teams": int(team_features[PREDICTION_COLUMNS].dropna().shape[0]),
        "data_gaps": [
            "No possession, shots, shots-on-target, or fouls dataset is present in the local project folder.",
            "The requested world_cup_2018_squads.xlsx file is not present; player-strength features are derived from FIFA18 nationality-level player records instead of exact 2018 roster membership.",
            "Jordan has ranking data but no FIFA18 player rows, so median player-strength features are used.",
        ],
    }
    joblib.dump(metrics, MODELS_DIR / "model_metrics.pkl")
    return metrics


if __name__ == "__main__":
    metrics = train_model()
    print("Saved model artifacts, datasets, and metrics")
    print(f"Best match model: {metrics['best_match_model']}")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Tournament-success rows: {metrics['tournament_success_rows']}")
    print(f"Prediction-ready teams: {metrics['available_prediction_teams']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
