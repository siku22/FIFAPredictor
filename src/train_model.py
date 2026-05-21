from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.model_selection import GridSearchCV, train_test_split
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

SIMULATION_MODEL_NAMES = [
    "Tuned Random Forest",
    "Logistic Regression",
    "SVM",
]

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
    "recent_win_rate_diff_5",
    "recent_goal_diff_diff_10",
    "recent_goals_for_diff_10",
    "recent_goals_against_diff_10",
    "recent_match_count_diff",
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
    "recent_win_rate_5",
    "recent_goal_diff_10",
    "recent_goals_for_10",
    "recent_goals_against_10",
    "recent_matches",
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
    "recent_win_rate_diff_5",
    "recent_goal_diff_diff_10",
    "recent_goals_for_diff_10",
    "recent_goals_against_diff_10",
    "recent_match_count_diff",
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


def prepare_rankings(rankings):
    rankings = rankings.rename(
        columns={
            "total.points": "total_points",
            "previous.points": "previous_points",
            "diff.points": "diff_points",
        }
    ).copy()
    rankings["team"] = rankings["team"].replace(TEAM_NAME_ALIASES)
    rankings["date"] = pd.to_datetime(rankings["date"].astype(str), format="%Y", errors="coerce")
    rankings = rankings.dropna(subset=["date", "team", "rank", "total_points"])
    return rankings.sort_values(["team", "date"])


def build_form_history(matches, names):
    name_map = dict(zip(names["former"], names["current"]))
    matches = matches.copy()
    matches["home_team"] = matches["home_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["away_team"] = matches["away_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    home_rows = matches[
        ["date", "home_team", "away_team", "home_goals", "away_goals"]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
        }
    )
    away_rows = matches[
        ["date", "away_team", "home_team", "away_goals", "home_goals"]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
        }
    )
    form = pd.concat([home_rows, away_rows], ignore_index=True).sort_values(["team", "date"])
    form["win"] = (form["goals_for"] > form["goals_against"]).astype(int)
    form["goal_diff"] = form["goals_for"] - form["goals_against"]

    grouped = form.groupby("team", group_keys=False)
    form["recent_win_rate_5"] = grouped["win"].apply(lambda values: values.shift().rolling(5, min_periods=1).mean())
    form["recent_goal_diff_10"] = grouped["goal_diff"].apply(lambda values: values.shift().rolling(10, min_periods=1).mean())
    form["recent_goals_for_10"] = grouped["goals_for"].apply(lambda values: values.shift().rolling(10, min_periods=1).mean())
    form["recent_goals_against_10"] = grouped["goals_against"].apply(
        lambda values: values.shift().rolling(10, min_periods=1).mean()
    )
    form["recent_matches"] = grouped.cumcount()
    return form


def latest_form_features(form_history):
    form_columns = [
        "recent_win_rate_5",
        "recent_goal_diff_10",
        "recent_goals_for_10",
        "recent_goals_against_10",
        "recent_matches",
    ]
    latest = form_history.sort_values(["team", "date"]).groupby("team", as_index=False).tail(1)
    return latest[["team", *form_columns]]


def build_team_features(players, rankings, form_history=None):
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

    rankings = prepare_rankings(rankings)
    latest_rankings = rankings.sort_values("date").groupby("team", as_index=False).tail(1)

    team_features = team_features.merge(
        latest_rankings[["team", "rank", "total_points", "previous_points", "diff_points"]],
        on="team",
        how="left",
    )
    if form_history is not None:
        team_features = team_features.merge(latest_form_features(form_history), on="team", how="left")

    form_columns = [
        "recent_win_rate_5",
        "recent_goal_diff_10",
        "recent_goals_for_10",
        "recent_goals_against_10",
        "recent_matches",
    ]
    for column in form_columns:
        if column not in team_features:
            team_features[column] = pd.NA
    team_features[form_columns] = team_features[form_columns].fillna(
        team_features[form_columns].median(numeric_only=True)
    )

    median_features = team_features[[column for column in PREDICTION_COLUMNS if column not in {"rank", "total_points"}]].median(
        numeric_only=True
    )
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


def ranking_as_of_matches(match_rows, rankings, side):
    lookup = rankings[["date", "team", "rank", "total_points", "previous_points", "diff_points"]].copy()
    rows = []
    team_column = f"{side}_team"
    for team, group in match_rows[[team_column, "date"]].dropna().groupby(team_column):
        history = lookup[lookup["team"] == team].sort_values("date")
        if history.empty:
            continue
        group = group.reset_index(names="_row_id")
        merged = pd.merge_asof(
            group.sort_values("date"),
            history,
            on="date",
            direction="backward",
        )
        rows.append(merged.set_index("_row_id").drop(columns=["team"]).rename(columns={team_column: "match_team"}))
    if not rows:
        return pd.DataFrame(index=match_rows.index)
    ranked = pd.concat(rows).sort_index()
    ranked = ranked.rename(
        columns={
            "rank": f"{side}_rank",
            "total_points": f"{side}_total_points",
            "previous_points": f"{side}_previous_points",
            "diff_points": f"{side}_diff_points",
        }
    )
    return ranked[[f"{side}_rank", f"{side}_total_points", f"{side}_previous_points", f"{side}_diff_points"]]


def form_as_of_matches(match_rows, form_history, side):
    form_columns = [
        "recent_win_rate_5",
        "recent_goal_diff_10",
        "recent_goals_for_10",
        "recent_goals_against_10",
        "recent_matches",
    ]
    rows = []
    team_column = f"{side}_team"
    for team, group in match_rows[[team_column, "date"]].dropna().groupby(team_column):
        history = form_history[form_history["team"] == team].sort_values("date")
        if history.empty:
            continue
        group = group.reset_index(names="_row_id")
        merged = pd.merge_asof(
            group.sort_values("date"),
            history[["date", *form_columns]],
            on="date",
            direction="backward",
        )
        rows.append(merged.set_index("_row_id").rename(columns={team_column: "match_team"}))
    if not rows:
        return pd.DataFrame(index=match_rows.index)
    form_rows = pd.concat(rows).sort_index()
    return form_rows[form_columns].add_prefix(f"{side}_")


def build_match_dataset(matches, names, team_features, rankings, form_history):
    name_map = dict(zip(names["former"], names["current"]))
    matches = matches.copy()
    matches["home_team"] = matches["home_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["away_team"] = matches["away_team"].replace(name_map).replace(TEAM_NAME_ALIASES)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
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

    rankings = prepare_rankings(rankings)
    rank_columns = ["rank", "total_points", "previous_points", "diff_points"]
    form_columns = [
        "recent_win_rate_5",
        "recent_goal_diff_10",
        "recent_goals_for_10",
        "recent_goals_against_10",
        "recent_matches",
    ]
    compiled = compiled.drop(
        columns=[f"{side}_{column}" for side in ("home", "away") for column in [*rank_columns, *form_columns]],
        errors="ignore",
    )
    compiled = compiled.join(ranking_as_of_matches(compiled, rankings, "home"))
    compiled = compiled.join(ranking_as_of_matches(compiled, rankings, "away"))
    compiled = compiled.join(form_as_of_matches(compiled, form_history, "home"))
    compiled = compiled.join(form_as_of_matches(compiled, form_history, "away"))

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
    compiled["recent_win_rate_diff_5"] = compiled["home_recent_win_rate_5"] - compiled["away_recent_win_rate_5"]
    compiled["recent_goal_diff_diff_10"] = compiled["home_recent_goal_diff_10"] - compiled["away_recent_goal_diff_10"]
    compiled["recent_goals_for_diff_10"] = compiled["home_recent_goals_for_10"] - compiled["away_recent_goals_for_10"]
    compiled["recent_goals_against_diff_10"] = (
        compiled["home_recent_goals_against_10"] - compiled["away_recent_goals_against_10"]
    )
    compiled["recent_match_count_diff"] = compiled["home_recent_matches"] - compiled["away_recent_matches"]
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
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    if y_prob is not None:
        confidence = np.maximum(y_prob, 1 - y_prob)
        high_confidence = confidence >= 0.70
        metrics["high_confidence_accuracy_70"] = (
            accuracy_score(y_test[high_confidence], y_pred[high_confidence])
            if high_confidence.any()
            else None
        )
        metrics["high_confidence_coverage_70"] = float(high_confidence.mean())
        metrics["high_confidence_rows_70"] = int(high_confidence.sum())
    return metrics


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
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "Tuned Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "n_estimators": [300],
                "max_depth": [8, 14],
                "min_samples_leaf": [3, 6],
            },
            cv=3,
            scoring="accuracy",
            n_jobs=1,
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
            "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
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
    form_history = build_form_history(matches, names)
    team_features = build_team_features(players, rankings, form_history)
    model_dataset = build_match_dataset(matches, names, team_features, rankings, form_history)

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
    joblib.dump(
        {name: trained_models[name] for name in SIMULATION_MODEL_NAMES if name in trained_models},
        MODELS_DIR / "match_models.pkl",
    )

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
            "Current/future predictions use each team's latest available rolling-form values from the local match history.",
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
