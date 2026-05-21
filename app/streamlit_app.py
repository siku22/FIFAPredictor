import base64
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REFERENCE_DATA_DIR = ROOT_DIR / "data" / "reference"
DOCS_DIR = ROOT_DIR / "docs"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
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
TEAM_COMPARISON_COLUMNS = {
    "avg_overall": "Overall",
    "attack_strength": "Attack",
    "midfield_strength": "Midfield",
    "defense_strength": "Defense",
    "goalkeeper_strength": "Goalkeeper",
    "avg_age": "Avg age",
    "team_balance": "Balance",
    "num_players": "Player pool",
    "elite_players": "Elite players",
    "rank": "FIFA rank",
    "total_points": "FIFA points",
    "recent_win_rate_5": "Recent win rate",
    "recent_goal_diff_10": "Recent goal diff",
    "recent_goals_for_10": "Recent goals for",
    "recent_goals_against_10": "Recent goals against",
    "recent_matches": "Recent sample",
}
FEATURE_LABELS = {
    "overall_diff": "overall rating",
    "attack_diff": "attack",
    "midfield_diff": "midfield",
    "defense_diff": "defense",
    "goalkeeper_diff": "goalkeeping",
    "age_diff": "average age",
    "balance_diff": "squad balance",
    "player_pool_diff": "player pool",
    "elite_player_diff": "elite-player count",
    "rank_advantage": "FIFA rank",
    "points_diff": "ranking points",
    "strength_diff": "combined strength",
    "spine_diff": "team spine",
    "rank_points_blend": "rank-points blend",
    "attack_vs_defense_gap": "attack-defense gap",
    "abs_strength_gap": "strength gap",
    "recent_win_rate_diff_5": "recent win rate",
    "recent_goal_diff_diff_10": "recent goal difference",
    "recent_goals_for_diff_10": "recent scoring form",
    "recent_goals_against_diff_10": "recent defensive form",
    "recent_match_count_diff": "recent match sample",
}
COUNTRY_THEMES = {
    "Brazil": {"primary": "#009739", "secondary": "#FEDD00", "accent": "#012169"},
    "France": {"primary": "#1A2A6C", "secondary": "#FFFFFF", "accent": "#EF4135"},
    "Argentina": {"primary": "#75AADB", "secondary": "#FFFFFF", "accent": "#FCBF49"},
    "Spain": {"primary": "#AA151B", "secondary": "#F1BF00", "accent": "#7A0C12"},
    "United States": {"primary": "#3C3B6E", "secondary": "#FFFFFF", "accent": "#B22234"},
    "Mexico": {"primary": "#006847", "secondary": "#FFFFFF", "accent": "#CE1126"},
    "Portugal": {"primary": "#006600", "secondary": "#FF0000", "accent": "#FFD100"},
    "Japan": {"primary": "#FFFFFF", "secondary": "#BC002D", "accent": "#1F2937"},
}
TEAM_FLAG_EMOJI = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼",
    "Czechia": "🇨🇿",
    "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵",
    "Jordan": "🇯🇴",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Türkiye": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}
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
SIMULATION_MODEL_NAMES = [
    "Tuned Random Forest",
    "Logistic Regression",
    "SVM",
]


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "fifa_model.pkl"
    match_models_path = MODELS_DIR / "match_models.pkl"
    features_path = MODELS_DIR / "team_features.pkl"
    metrics_path = MODELS_DIR / "model_metrics.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python3 src/train_model.py` first."
        )

    metrics = joblib.load(metrics_path) if metrics_path.exists() else {}
    model = joblib.load(model_path)
    match_models = joblib.load(match_models_path) if match_models_path.exists() else {}
    if not match_models:
        match_models = {metrics.get("best_match_model", "Best Model"): model}
    return model, match_models, joblib.load(features_path), metrics


def prediction_ready_teams(team_features):
    ready = team_features.dropna(subset=PREDICTION_COLUMNS)
    return sorted(ready["team"].dropna().unique())


def qualified_prediction_teams(team_features):
    ready = set(prediction_ready_teams(team_features))
    return [team for team in QUALIFIED_2026_TEAMS if team in ready]


def metrics_table(results, columns):
    rows = []
    for model_name, values in results.items():
        row = {"Model": model_name}
        for column in columns:
            value = values.get(column)
            row[column.upper() if column != "roc_auc" else "ROC-AUC"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_matrix_table(matrix):
    if not matrix:
        return pd.DataFrame()
    return pd.DataFrame(
        matrix,
        index=["Actual no home win", "Actual home win"],
        columns=["Predicted no home win", "Predicted home win"],
    )


def initials(team_name):
    parts = [part for part in team_name.replace("&", " ").split() if part]
    if not parts:
        return "FC"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def team_theme(team_name):
    if team_name in COUNTRY_THEMES:
        return COUNTRY_THEMES[team_name]
    palette = [
        {"primary": "#0B5D3B", "secondary": "#D6B35A", "accent": "#063823"},
        {"primary": "#1A2A6C", "secondary": "#FFFFFF", "accent": "#EF4135"},
        {"primary": "#AA151B", "secondary": "#F1BF00", "accent": "#7A0C12"},
        {"primary": "#006847", "secondary": "#FFFFFF", "accent": "#CE1126"},
        {"primary": "#552583", "secondary": "#FDB927", "accent": "#2C1A4D"},
        {"primary": "#0057B8", "secondary": "#FFD700", "accent": "#183153"},
    ]
    return palette[sum(ord(char) for char in team_name) % len(palette)]


def team_badge(team_name, compact=False):
    theme = team_theme(team_name)
    flag = TEAM_FLAG_EMOJI.get(team_name, "")
    label = initials(team_name) if compact else team_name
    return (
        f'<span class="team-badge" style="--badge-primary:{theme["primary"]};'
        f'--badge-secondary:{theme["secondary"]};--badge-accent:{theme["accent"]};">'
        f'<span class="badge-mark">{flag or initials(team_name)}</span>'
        f'<span>{label}</span></span>'
    )


def badge_line(teams):
    return "".join(team_badge(team) for team in teams)


def team_row(team_name, team_features):
    tf = team_features.copy()
    tf["team_clean"] = tf["team"].str.lower().str.strip()
    return tf[tf["team_clean"] == team_name.lower().strip()].iloc[0]


def team_comparison_table(team_a, team_b, team_features):
    row_a = team_row(team_a, team_features)
    row_b = team_row(team_b, team_features)
    rows = []
    for column, label in TEAM_COMPARISON_COLUMNS.items():
        value_a = row_a[column]
        value_b = row_b[column]
        if column == "rank":
            leader = team_a if value_a < value_b else team_b if value_b < value_a else "Even"
        else:
            leader = team_a if value_a > value_b else team_b if value_b > value_a else "Even"
        rows.append(
            {
                "Metric": label,
                team_a: value_a,
                team_b: value_b,
                "Advantage": leader,
            }
        )
    comparison = pd.DataFrame(rows)
    numeric_columns = [team_a, team_b]
    comparison[numeric_columns] = comparison[numeric_columns].round(2)
    return comparison


def confidence_label(prob_a, prob_b):
    gap = abs(prob_a - prob_b)
    if gap >= 0.3:
        return "High", gap
    if gap >= 0.15:
        return "Medium", gap
    return "Low", gap


def prediction_explanation(result, input_data, team_a, team_b):
    winner = result["winner"]
    loser = team_b if winner == team_a else team_a
    row = input_data.iloc[0]
    winner_sign = 1 if winner == team_a else -1
    advantages = []
    for feature in FEATURES:
        value = row[feature] * winner_sign
        if feature == "balance_diff":
            value = -value
        if value > 0:
            advantages.append((FEATURE_LABELS[feature], abs(row[feature])))

    top_advantages = [label for label, _ in sorted(advantages, key=lambda item: item[1], reverse=True)[:3]]
    if not top_advantages:
        return f"The model favors {winner}, but the feature differences are fairly balanced against {loser}."
    if len(top_advantages) == 1:
        factors = top_advantages[0]
    else:
        factors = ", ".join(top_advantages[:-1]) + f", and {top_advantages[-1]}"
    return f"The model favors {winner} mainly because of advantages in {factors} relative to {loser}."


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


def format_percent_table(df):
    formatted = df.copy()
    for column in formatted.columns:
        if column != "Team":
            formatted[column] = formatted[column].map("{:.2%}".format)
    return formatted


def champion_explanation(champion_name, team_features):
    row = team_row(champion_name, team_features)
    ready = team_features.dropna(subset=PREDICTION_COLUMNS)
    strengths = []
    comparisons = {
        "attack": ("attack_strength", "attack strength", True),
        "midfield": ("midfield_strength", "midfield strength", True),
        "defense": ("defense_strength", "defensive strength", True),
        "goalkeeper": ("goalkeeper_strength", "goalkeeper rating", True),
        "points": ("total_points", "FIFA ranking points", True),
        "rank": ("rank", "FIFA rank", False),
    }
    for _, (column, label, higher_is_better) in comparisons.items():
        value = row[column]
        median = ready[column].median()
        if higher_is_better and value >= median:
            strengths.append((label, abs(value - median)))
        elif not higher_is_better and value <= median:
            strengths.append((label, abs(value - median)))

    top = [label for label, _ in sorted(strengths, key=lambda item: item[1], reverse=True)[:3]]
    if not top:
        return f"{champion_name} rises through the simulation mostly through balanced probabilities across rounds."
    if len(top) == 1:
        factors = top[0]
    else:
        factors = ", ".join(top[:-1]) + f", and {top[-1]}"
    return f"{champion_name} grades well because its team profile is strong in {factors} compared with the available field."


def render_bracket(sample_result):
    if not sample_result:
        return
    rounds = [
        ("Round of 32", sample_result.get("round_of_32", [])),
        ("Round of 16", sample_result.get("round_of_16", [])),
        ("Quarterfinals", sample_result.get("quarterfinals", [])),
        ("Semifinals", sample_result.get("semifinals", [])),
        ("Finalists", sample_result.get("finalists", [])),
        ("Champion", [sample_result.get("champion", "")]),
    ]
    columns = []
    for round_name, teams in rounds:
        team_markup = "".join(
            f'<div class="bracket-team">{team_badge(team, compact=True)}</div>'
            for team in teams
            if team
        )
        columns.append(
            f'<div class="bracket-round"><div class="bracket-title">{round_name}</div>{team_markup}</div>'
        )
    st.markdown(f'<div class="bracket-board">{"".join(columns)}</div>', unsafe_allow_html=True)


def add_file_download(label, path, mime):
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
        )


def reset_penalty_game():
    st.session_state["penalty_score"] = 0
    st.session_state["penalty_kicks"] = 0
    st.session_state["penalty_streak"] = 0
    st.session_state["penalty_best_streak"] = 0
    st.session_state["penalty_history"] = []
    st.session_state["penalty_message"] = "Pick a corner and take the first shot."
    st.session_state["penalty_last_shot"] = "idle"
    st.session_state["penalty_last_keeper"] = "center"
    st.session_state["penalty_last_result"] = "ready"
    st.session_state["penalty_crowd"] = 50
    st.session_state["penalty_celebrate"] = False


def play_penalty(direction, shot_style, power, difficulty):
    if "penalty_score" not in st.session_state:
        reset_penalty_game()
    if st.session_state["penalty_kicks"] >= 5:
        return

    keeper_weights = {
        "Friendly": [0.32, 0.22, 0.32],
        "Balanced": [0.36, 0.28, 0.36],
        "Final Boss": [0.41, 0.34, 0.41],
    }
    weights = np.array(keeper_weights[difficulty], dtype=float)
    if direction == "Left":
        weights[0] += 0.08
    elif direction == "Center":
        weights[1] += 0.08
    else:
        weights[2] += 0.08
    weights = weights / weights.sum()

    keeper_direction = np.random.choice(["Left", "Center", "Right"], p=weights)
    style_save_bonus = {"Placement": -0.08, "Power": 0.02, "Chip": -0.02}
    style_miss_bonus = {"Placement": 0.02, "Power": 0.08, "Chip": 0.05}
    difficulty_save_bonus = {"Friendly": -0.08, "Balanced": 0.0, "Final Boss": 0.1}
    miss_chance = max(0.01, min(0.22, abs(power - 6) * 0.018 + style_miss_bonus[shot_style]))
    save_chance = 0.82 + style_save_bonus[shot_style] + difficulty_save_bonus[difficulty]
    same_direction = direction == keeper_direction
    missed = np.random.random() < miss_chance
    saved = same_direction and np.random.random() < save_chance
    scored = not missed and not saved

    st.session_state["penalty_kicks"] += 1
    if scored:
        st.session_state["penalty_score"] += 1
        st.session_state["penalty_streak"] += 1
        st.session_state["penalty_best_streak"] = max(
            st.session_state["penalty_best_streak"],
            st.session_state["penalty_streak"],
        )
        st.session_state["penalty_crowd"] = min(100, st.session_state["penalty_crowd"] + 14)
        st.session_state["penalty_celebrate"] = True
    else:
        st.session_state["penalty_streak"] = 0
        st.session_state["penalty_crowd"] = max(5, st.session_state["penalty_crowd"] - 10)

    result = "Goal" if scored else "Wide" if missed else "Saved"
    st.session_state["penalty_message"] = (
        f"{result}. {shot_style} shot, power {power}/10. "
        f"You aimed {direction.lower()} and the keeper went {keeper_direction.lower()}."
    )
    st.session_state["penalty_last_shot"] = direction.lower()
    st.session_state["penalty_last_keeper"] = keeper_direction.lower()
    st.session_state["penalty_last_result"] = result.lower()
    st.session_state["penalty_history"].append(
        {
            "Kick": st.session_state["penalty_kicks"],
            "Shot": direction,
            "Style": shot_style,
            "Power": power,
            "Keeper": keeper_direction,
            "Result": result,
        }
    )


def render_penalty_game():
    if "penalty_score" not in st.session_state:
        reset_penalty_game()

    kicks = st.session_state["penalty_kicks"]
    score = st.session_state["penalty_score"]
    remaining = max(0, 5 - kicks)
    streak = st.session_state.get("penalty_streak", 0)
    best_streak = st.session_state.get("penalty_best_streak", 0)
    crowd = st.session_state.get("penalty_crowd", 50)
    shot_class = st.session_state.get("penalty_last_shot", "idle")
    keeper_class = st.session_state.get("penalty_last_keeper", "center")
    result_class = st.session_state.get("penalty_last_result", "ready")
    result_text = "Ready" if result_class == "ready" else result_class.title()
    team_name = st.session_state.get("penalty_team_name", "Tiny Strikers").strip() or "Tiny Strikers"

    if st.session_state.get("penalty_celebrate"):
        st.balloons()
        st.session_state["penalty_celebrate"] = False

    control_a, control_b, control_c = st.columns([1.2, 1, 1])
    with control_a:
        team_name = st.text_input("Team name", value=team_name, key="penalty_team_name").strip() or "Tiny Strikers"
    with control_b:
        selected_theme = st.selectbox("Country color theme", list(COUNTRY_THEMES), key="penalty_country_theme")
    with control_c:
        difficulty = st.selectbox("Keeper difficulty", ["Friendly", "Balanced", "Final Boss"], index=1)
    theme = COUNTRY_THEMES[selected_theme]
    primary = theme["primary"]
    secondary = theme["secondary"]
    accent = theme["accent"]

    style_col, power_col = st.columns([1, 2])
    with style_col:
        shot_style = st.selectbox("Shot style", ["Placement", "Power", "Chip"], index=0)
    with power_col:
        power = st.slider("Shot power", min_value=1, max_value=10, value=6)

    st.markdown(
        f"""
        <div class="game-pitch themed-pitch" style="
            --team-primary:{primary};
            --team-secondary:{secondary};
            --team-accent:{accent};
            background: linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px),
                        radial-gradient(circle at 84% 14%, {secondary}55, transparent 24%),
                        linear-gradient(135deg, {primary}, {accent});
            border-color:{secondary};
        ">
            <div class="eyebrow">Mini game</div>
            <h3 style="margin:0;color:white;">Penalty Shootout</h3>
            <div style="color:rgba(255,255,255,0.82);margin-top:0.35rem;">
                {team_name} get five kicks. Pick a style, set the power, and beat the keeper.
            </div>
            <div class="mini-scoreboard">
                <span class="kit-swatch">
                    <i style="background:{primary};"></i>
                    <i style="background:{secondary};"></i>
                    <i style="background:{accent};"></i>
                </span>
                <span>Theme: <strong>{selected_theme}</strong></span>
                <span>Team: <strong>{team_name}</strong></span>
                <span>Best streak: <strong>{best_streak}</strong></span>
                <span>Crowd: <strong>{crowd}%</strong></span>
            </div>
            <div class="animated-goal result-{result_class}" style="border-color:{secondary}; box-shadow: inset 0 0 0 4px {accent}66;">
                <div class="net-lines"></div>
                <div class="keeper keeper-{keeper_class}" style="background:{secondary}; border-color:{accent};">GK</div>
                <div class="ball ball-{shot_class}" style="background:{secondary}; border-color:{accent};"></div>
                <div class="result-burst">{result_text}</div>
                <div class="goal-cell" style="background:{primary}33;">Left</div>
                <div class="goal-cell" style="background:{secondary}22;">Center</div>
                <div class="goal-cell" style="background:{accent}33;">Right</div>
            </div>
            <div class="shot-lane" style="border-color:{secondary}88;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(crowd / 100, text="Crowd energy")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Goals", f"{score}/5")
    with g2:
        st.metric("Kicks Taken", kicks)
    with g3:
        st.metric("Streak", streak, f"{remaining} kicks left")

    st.markdown(f'<div class="game-status">{st.session_state["penalty_message"]}</div>', unsafe_allow_html=True)

    shot_cols = st.columns(3)
    disabled = kicks >= 5
    with shot_cols[0]:
        if st.button("Shoot Left", use_container_width=True, disabled=disabled):
            play_penalty("Left", shot_style, power, difficulty)
            st.rerun()
    with shot_cols[1]:
        if st.button("Shoot Center", use_container_width=True, disabled=disabled):
            play_penalty("Center", shot_style, power, difficulty)
            st.rerun()
    with shot_cols[2]:
        if st.button("Shoot Right", use_container_width=True, disabled=disabled):
            play_penalty("Right", shot_style, power, difficulty)
            st.rerun()

    if kicks >= 5:
        if score >= 4:
            st.success("Final whistle: elite finishing. The crowd is singing your team name.")
        elif score >= 2:
            st.info("Final whistle: respectable shootout. Solid work from the spot.")
        else:
            st.warning("Final whistle: the keeper had your number. Time for extra shooting practice.")

    if st.button("Reset Shootout", use_container_width=True):
        reset_penalty_game()
        st.rerun()

    history = pd.DataFrame(st.session_state["penalty_history"])
    if not history.empty:
        st.subheader("Shot Log")
        st.dataframe(history, use_container_width=True, hide_index=True)


def inject_soccer_theme(theme_mode="Stadium Mode"):
    clean_mode_css = ""
    if theme_mode == "Clean Report Mode":
        clean_mode_css = """
        .stApp {
            background: #f4f7f5;
            color: #102017;
        }
        [data-testid="stHeader"] {
            background: rgba(244,247,245,0.9);
        }
        .hero-panel {
            background: linear-gradient(135deg, #ffffff, #eaf5ee);
            border-color: rgba(11,93,59,0.18);
            box-shadow: 0 12px 34px rgba(16,32,23,0.12);
        }
        .hero-title {
            color: #0b5d3b;
        }
        .hero-copy {
            color: #496256;
        }
        [data-testid="stTabs"] button {
            color: #102017;
        }
        """
    css = """
        <style>
        :root {
            --pitch: #0b5d3b;
            --pitch-dark: #063823;
            --line: rgba(255, 255, 255, 0.18);
            --gold: #d6b35a;
            --mint: #78d6a3;
            --ink: #102017;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
                linear-gradient(0deg, rgba(255,255,255,0.03) 1px, transparent 1px),
                radial-gradient(circle at 50% 12%, rgba(214,179,90,0.12), transparent 24%),
                linear-gradient(135deg, #08351f 0%, #0a5236 52%, #062f22 100%);
            background-size: 96px 96px, 96px 96px, auto, auto;
            color: #f7fbf6;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stTabs"] {
            margin-top: 0.6rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.45rem;
            padding: 0.4rem;
            margin-bottom: 1.25rem;
            background: rgba(255,255,255,0.11);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
        }

        [data-testid="stTabs"] button,
        [data-testid="stTabs"] [role="tab"] {
            color: #ecfff4;
            border-radius: 6px !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
            padding: 0.55rem 0.9rem;
        }

        [data-testid="stTabs"] button[aria-selected="true"],
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: rgba(255,255,255,0.18);
            color: #ffffff;
            border: 1px solid rgba(214,179,90,0.62) !important;
        }

        [data-testid="stTabs"] [role="tab"] p {
            margin: 0;
        }

        .stButton > button[kind="primary"] {
            background: #d6b35a;
            color: #102017;
            border: 1px solid rgba(255,255,255,0.35);
            font-weight: 900;
        }

        .stButton > button[kind="secondary"] {
            border: 1px solid rgba(214,179,90,0.45);
        }

        .stButton {
            margin: 0.65rem 0 1.35rem;
        }

        .stButton > button {
            border-radius: 8px !important;
            min-height: 2.65rem;
            box-shadow: 0 6px 14px rgba(0,0,0,0.1);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink);
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: clamp(1.45rem, 5vw, 2.25rem);
            line-height: 1.08;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        [data-testid="stMetric"] {
            min-height: 116px;
            margin-bottom: 1.1rem;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stExpander"],
        div[data-testid="stAlert"],
        div[data-testid="stForm"] {
            border-radius: 8px;
        }

        .hero-panel {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            padding: 2rem;
            border: 1px solid rgba(255,255,255,0.22);
            background:
                linear-gradient(90deg, rgba(255,255,255,0.13) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.08), rgba(255,255,255,0.02)),
                linear-gradient(135deg, rgba(10,88,54,0.95), rgba(5,48,32,0.95));
            box-shadow: 0 22px 60px rgba(0,0,0,0.24);
        }

        .hero-panel:before {
            content: "";
            position: absolute;
            inset: 18px;
            border: 2px solid rgba(255,255,255,0.22);
            border-radius: 8px;
            pointer-events: none;
        }

        .hero-panel:after {
            content: "";
            position: absolute;
            display: none;
        }

        .hero-ball {
            position: absolute;
            top: 50%;
            right: 8%;
            width: 172px;
            height: 172px;
            transform: translateY(-50%);
            pointer-events: none;
            opacity: 0.92;
            filter: drop-shadow(0 18px 34px rgba(0,0,0,0.28));
            z-index: 1;
        }

        .hero-ball svg {
            width: 100%;
            height: 100%;
            display: block;
            animation: soccer-spin 7s linear infinite;
            transform-origin: 50% 50%;
        }

        @keyframes soccer-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .hero-content {
            position: relative;
            z-index: 2;
        }

        .eyebrow {
            color: var(--mint);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: clamp(2rem, 5vw, 4.1rem);
            line-height: 1;
            font-weight: 900;
            margin: 0;
            max-width: 780px;
        }

        .hero-copy {
            color: rgba(255,255,255,0.82);
            font-size: 1.05rem;
            max-width: 760px;
            margin-top: 1rem;
        }

        .stat-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1.2rem 0 1.4rem;
        }

        .stat-card,
        .scoreboard,
        .section-card,
        .chart-panel,
        .coach-note {
            background: rgba(255,255,255,0.96);
            color: var(--ink);
            border: 1px solid rgba(255,255,255,0.62);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        }

        .section-card {
            margin-bottom: 0.9rem;
        }

        .stat-label {
            color: #496256;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-value {
            font-size: 1.55rem;
            font-weight: 900;
            margin-top: 0.2rem;
        }

        .report-stat-card {
            min-height: 116px;
            background: rgba(255,255,255,0.94);
            color: var(--ink);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.16);
        }

        .report-stat-value {
            font-size: clamp(1.55rem, 3.4vw, 2.25rem);
            line-height: 1.08;
            font-weight: 900;
            margin-top: 0.55rem;
            overflow-wrap: anywhere;
            white-space: normal;
        }

        .scoreboard {
            text-align: center;
            margin: 2rem 0 1.35rem;
        }

        .score-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 1rem;
        }

        .team-name {
            font-size: clamp(1.25rem, 3vw, 2rem);
            font-weight: 900;
        }

        .vs-badge {
            background: var(--pitch);
            color: #ffffff;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            font-weight: 900;
            border: 2px solid var(--gold);
        }

        .winner-banner {
            margin-top: 1rem;
            background: linear-gradient(90deg, #0b5d3b, #13784f);
            color: white;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.25);
        }

        .group-pill {
            display: inline-block;
            margin: 0.18rem;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(11,93,59,0.1);
            color: #0c3d2a;
            border: 1px solid rgba(11,93,59,0.2);
            font-weight: 650;
            font-size: 0.88rem;
        }

        .team-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            margin: 0.16rem;
            padding: 0.32rem 0.6rem;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--badge-primary), var(--badge-accent));
            color: #ffffff;
            border: 1px solid color-mix(in srgb, var(--badge-secondary) 75%, white);
            font-weight: 800;
            box-shadow: 0 8px 18px rgba(0,0,0,0.14);
        }

        .badge-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.65rem;
            height: 1.65rem;
            border-radius: 999px;
            background: var(--badge-secondary);
            color: var(--badge-accent);
            font-weight: 900;
            border: 1px solid rgba(255,255,255,0.62);
        }

        .badge-row {
            margin: 0.45rem 0 0.8rem;
        }

        .chart-panel {
            min-height: 0;
            margin: 1rem 0 0.25rem;
            background: rgba(255,255,255,0.96);
            border-color: rgba(255,255,255,0.62);
            box-shadow: 0 8px 18px rgba(0,0,0,0.1);
        }

        .chart-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            color: var(--ink);
            font-weight: 900;
            margin-bottom: 0.25rem;
            font-size: 1.08rem;
        }

        .chart-hint,
        .coach-note {
            color: #496256;
            font-size: 0.92rem;
        }

        div[data-testid="stVegaLiteChart"] {
            background: #f7fbf6;
            border: 1px solid rgba(11,93,59,0.16);
            border-radius: 8px;
            padding: 0.7rem;
            margin: 0.25rem 0 1.55rem;
            box-shadow: 0 8px 18px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] {
            margin: 0.9rem 0 2.35rem;
            border: 1px solid rgba(11,93,59,0.16);
            box-shadow: 0 8px 18px rgba(0,0,0,0.1);
        }

        .chart-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            background: #0b5d3b;
            color: #ffffff;
            border: 1px solid var(--gold);
            font-size: 0.78rem;
            font-weight: 900;
            white-space: nowrap;
        }

        .coach-note {
            border-left: 5px solid var(--gold);
            margin: 0.8rem 0 1.25rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        }

        .coach-note strong {
            color: #0b5d3b;
        }

        .bracket-board {
            display: grid;
            grid-template-columns: repeat(6, minmax(130px, 1fr));
            gap: 0.7rem;
            overflow-x: auto;
            padding: 0.8rem;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(255,255,255,0.52);
            border-radius: 8px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.15);
        }

        .bracket-round {
            min-width: 130px;
            border-left: 3px solid #0b5d3b;
            padding-left: 0.55rem;
        }

        .bracket-title {
            color: #496256;
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .bracket-team {
            margin-bottom: 0.3rem;
        }

        .game-pitch {
            background:
                linear-gradient(90deg, rgba(255,255,255,0.12) 1px, transparent 1px),
                linear-gradient(135deg, rgba(11,93,59,0.96), rgba(6,56,35,0.96));
            background-size: 58px 58px, auto;
            border: 2px solid rgba(255,255,255,0.5);
            border-radius: 8px;
            padding: 1.2rem;
            color: white;
            box-shadow: 0 18px 40px rgba(0,0,0,0.18);
        }

        .themed-pitch {
            background:
                linear-gradient(90deg, rgba(255,255,255,0.12) 1px, transparent 1px),
                radial-gradient(circle at 82% 12%, color-mix(in srgb, var(--team-secondary) 34%, transparent), transparent 22%),
                linear-gradient(135deg, color-mix(in srgb, var(--team-primary) 82%, #063823), #063823);
            border-color: color-mix(in srgb, var(--team-secondary) 72%, white);
        }

        .goal-frame,
        .animated-goal {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin: 0.8rem 0;
            padding: 0.8rem;
            border: 3px solid rgba(255,255,255,0.78);
            border-bottom-width: 8px;
            border-radius: 8px 8px 4px 4px;
            background: rgba(255,255,255,0.1);
        }

        .animated-goal {
            position: relative;
            min-height: 230px;
            overflow: hidden;
            align-items: stretch;
            background:
                linear-gradient(rgba(255,255,255,0.18) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px),
                rgba(255,255,255,0.08);
            background-size: 28px 28px;
        }

        .themed-pitch .animated-goal {
            border-color: color-mix(in srgb, var(--team-secondary) 70%, white);
            box-shadow: inset 0 0 0 3px color-mix(in srgb, var(--team-accent) 28%, transparent);
        }

        .mini-scoreboard {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.8rem;
        }

        .mini-scoreboard span {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            color: #ffffff;
        }

        .kit-swatch {
            display: inline-flex;
            gap: 0.25rem;
            align-items: center;
        }

        .kit-swatch i {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.7);
            display: inline-block;
        }

        .kit-swatch i:nth-child(1) {
            background: var(--team-primary);
        }

        .kit-swatch i:nth-child(2) {
            background: var(--team-secondary);
        }

        .kit-swatch i:nth-child(3) {
            background: var(--team-accent);
        }

        .goal-cell {
            min-height: 190px;
            border: 1px dashed rgba(255,255,255,0.45);
            border-radius: 8px;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            text-align: center;
            font-weight: 800;
            color: #ffffff;
            padding-top: 0.65rem;
            z-index: 1;
        }

        .net-lines {
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 50% 110%, transparent 0 38%, rgba(255,255,255,0.2) 39%, transparent 40%),
                linear-gradient(90deg, transparent 32%, rgba(255,255,255,0.26) 33%, transparent 34%, transparent 65%, rgba(255,255,255,0.26) 66%, transparent 67%);
            pointer-events: none;
        }

        .keeper {
            position: absolute;
            left: 50%;
            bottom: 20px;
            width: 66px;
            height: 46px;
            transform: translateX(-50%);
            border-radius: 16px 16px 10px 10px;
            background: var(--team-secondary, #d6b35a);
            color: #102017;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            border: 2px solid var(--team-accent, rgba(255,255,255,0.8));
            z-index: 4;
            box-shadow: 0 10px 22px rgba(0,0,0,0.28);
            animation: keeper-ready 0.7s ease-in-out infinite alternate;
        }

        .keeper-left {
            animation: keeper-left 0.9s cubic-bezier(.2,.8,.2,1) both;
        }

        .keeper-center {
            animation: keeper-center 0.9s cubic-bezier(.2,.8,.2,1) both;
        }

        .keeper-right {
            animation: keeper-right 0.9s cubic-bezier(.2,.8,.2,1) both;
        }

        .ball {
            position: absolute;
            left: 50%;
            bottom: -18px;
            width: 34px;
            height: 34px;
            transform: translateX(-50%);
            border-radius: 50%;
            background:
                radial-gradient(circle at 35% 30%, #ffffff 0 18%, transparent 19%),
                radial-gradient(circle at 64% 68%, #111 0 10%, transparent 11%),
                var(--team-secondary, #f7fbf6);
            border: 2px solid var(--team-accent, #102017);
            z-index: 5;
            box-shadow: 0 12px 18px rgba(0,0,0,0.28);
        }

        .ball-idle {
            animation: ball-idle 1s ease-in-out infinite alternate;
        }

        .ball-left {
            animation: ball-left 0.9s cubic-bezier(.2,.9,.2,1) both;
        }

        .ball-center {
            animation: ball-center 0.9s cubic-bezier(.2,.9,.2,1) both;
        }

        .ball-right {
            animation: ball-right 0.9s cubic-bezier(.2,.9,.2,1) both;
        }

        .result-burst {
            position: absolute;
            left: 50%;
            top: 44%;
            transform: translate(-50%, -50%) scale(0.8);
            opacity: 0;
            z-index: 6;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(16,32,23,0.82);
            color: white;
            border: 1px solid rgba(255,255,255,0.4);
            font-weight: 900;
            letter-spacing: 0.03em;
            animation: result-pop 1.1s ease-out both;
        }

        .result-ready .result-burst {
            animation: none;
            opacity: 0;
        }

        .result-goal .result-burst {
            background: #0b5d3b;
        }

        .result-saved .result-burst {
            background: #7a2f20;
        }

        .result-wide .result-burst {
            background: #7a5b20;
        }

        .shot-lane {
            width: 44%;
            height: 72px;
            margin: 0 auto;
            border-left: 2px solid rgba(255,255,255,0.28);
            border-right: 2px solid rgba(255,255,255,0.28);
            border-bottom: 2px solid rgba(255,255,255,0.22);
            border-radius: 0 0 120px 120px;
        }

        @keyframes ball-idle {
            from { transform: translateX(-50%) translateY(0); }
            to { transform: translateX(-50%) translateY(-8px); }
        }

        @keyframes ball-left {
            0% { left: 50%; bottom: -18px; transform: translateX(-50%) scale(1); }
            70% { left: 18%; bottom: 150px; transform: translateX(-50%) scale(0.78) rotate(-260deg); }
            100% { left: 18%; bottom: 150px; transform: translateX(-50%) scale(0.78) rotate(-300deg); }
        }

        @keyframes ball-center {
            0% { left: 50%; bottom: -18px; transform: translateX(-50%) scale(1); }
            70% { left: 50%; bottom: 160px; transform: translateX(-50%) scale(0.78) rotate(220deg); }
            100% { left: 50%; bottom: 160px; transform: translateX(-50%) scale(0.78) rotate(260deg); }
        }

        @keyframes ball-right {
            0% { left: 50%; bottom: -18px; transform: translateX(-50%) scale(1); }
            70% { left: 82%; bottom: 150px; transform: translateX(-50%) scale(0.78) rotate(260deg); }
            100% { left: 82%; bottom: 150px; transform: translateX(-50%) scale(0.78) rotate(300deg); }
        }

        @keyframes keeper-ready {
            from { transform: translateX(-50%) translateY(0); }
            to { transform: translateX(-50%) translateY(-4px); }
        }

        @keyframes keeper-left {
            0% { left: 50%; bottom: 20px; transform: translateX(-50%) rotate(0deg); }
            100% { left: 21%; bottom: 108px; transform: translateX(-50%) rotate(-22deg); }
        }

        @keyframes keeper-center {
            0% { left: 50%; bottom: 20px; transform: translateX(-50%) scale(1); }
            100% { left: 50%; bottom: 112px; transform: translateX(-50%) scale(1.08); }
        }

        @keyframes keeper-right {
            0% { left: 50%; bottom: 20px; transform: translateX(-50%) rotate(0deg); }
            100% { left: 79%; bottom: 108px; transform: translateX(-50%) rotate(22deg); }
        }

        @keyframes result-pop {
            0%, 45% { opacity: 0; transform: translate(-50%, -50%) scale(0.75); }
            62% { opacity: 1; transform: translate(-50%, -50%) scale(1.08); }
            100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }

        .game-status {
            background: rgba(255,255,255,0.94);
            color: var(--ink);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.5);
            margin: 1.25rem 0 1rem;
        }

        @media (max-width: 780px) {
            .stat-strip,
            .score-row {
                grid-template-columns: 1fr;
            }
            .bracket-board {
                grid-template-columns: repeat(6, 150px);
            }
            .hero-panel {
                padding: 1.4rem;
            }
            .hero-ball {
                width: 92px;
                height: 92px;
                right: 1.4rem;
                top: 1.4rem;
                transform: none;
                opacity: 0.5;
            }
            .hero-title,
            .hero-copy {
                max-width: 100%;
            }
        }
        __CLEAN_MODE_CSS__
        </style>
        """.replace("__CLEAN_MODE_CSS__", clean_mode_css)
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def render_hero(metrics):
    ball_path = ROOT_DIR / "app" / "assets" / "soccer-ball.png"
    ball_src = ""
    if ball_path.exists():
        ball_src = "data:image/png;base64," + base64.b64encode(ball_path.read_bytes()).decode("ascii")

    components.html(
        f"""
        <style>
            :root {{
                --pitch: #0b5d3b;
                --gold: #d6b35a;
                --mint: #78d6a3;
                --ink: #102017;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background: transparent;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .hero-panel {{
                position: relative;
                overflow: hidden;
                border-radius: 8px;
                min-height: 238px;
                padding: 1.8rem 2rem;
                border: 1px solid rgba(255,255,255,0.22);
                background:
                    linear-gradient(90deg, rgba(255,255,255,0.13) 1px, transparent 1px),
                    linear-gradient(rgba(255,255,255,0.08), rgba(255,255,255,0.02)),
                    linear-gradient(135deg, rgba(10,88,54,0.95), rgba(5,48,32,0.95));
                box-shadow: 0 22px 60px rgba(0,0,0,0.24);
            }}
            .hero-panel:before {{
                content: "";
                position: absolute;
                inset: 18px;
                border: 2px solid rgba(255,255,255,0.22);
                border-radius: 8px;
                pointer-events: none;
            }}
            .hero-content {{
                position: relative;
                z-index: 2;
                max-width: calc(100% - 190px);
            }}
            .eyebrow {{
                color: var(--mint);
                font-size: 0.8rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }}
            .hero-title {{
                color: #ffffff;
                font-size: clamp(2rem, 5vw, 4.1rem);
                line-height: 1;
                font-weight: 900;
                margin: 0;
                max-width: 860px;
            }}
            .hero-copy {{
                color: rgba(255,255,255,0.82);
                font-size: 1.05rem;
                line-height: 1.55;
                max-width: 840px;
                margin-top: 1rem;
            }}
            .hero-ball {{
                position: absolute;
                top: 50%;
                right: 8%;
                width: 132px;
                height: 132px;
                transform: translateY(-50%);
                z-index: 3;
                user-select: none;
                filter: drop-shadow(0 18px 34px rgba(0,0,0,0.32));
                display: flex;
                align-items: center;
                justify-content: center;
                pointer-events: none;
            }}
            .ball-rotor {{
                width: 100%;
                height: 100%;
                transform-origin: 50% 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
            }}
            .ball-rotor img {{
                width: 100%;
                height: 100%;
                display: block;
                object-fit: contain;
                border-radius: 50%;
            }}
            @media (max-width: 780px) {{
                .hero-panel {{
                    min-height: 320px;
                    padding: 1.4rem;
                }}
                .hero-content {{
                    max-width: 100%;
                    padding-right: 0;
                }}
                .hero-ball {{
                    width: 92px;
                    height: 92px;
                    right: 1.4rem;
                    bottom: 1.15rem;
                    top: auto;
                    transform: none;
                    opacity: 0.8;
                }}
            }}
        </style>
        <div class="hero-panel">
            <div class="hero-content">
                <div class="eyebrow">World Cup analytics dashboard</div>
                <h1 class="hero-title">2026 FIFA World Cup Predictor</h1>
                <div class="hero-copy">
                    Match probabilities and tournament simulations powered by player ratings,
                    team composition, FIFA rankings, and historical international results.
                </div>
            </div>
            <div class="hero-ball" aria-label="Soccer ball">
                <div class="ball-rotor">
                    <img src="{ball_src}" alt="Soccer ball" />
                </div>
            </div>
        </div>
        """,
        height=258,
        scrolling=False,
    )
    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat-card">
                <div class="stat-label">Best Model</div>
                <div class="stat-value">{metrics.get("best_match_model", "N/A")}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">{metrics.get("accuracy", 0):.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Qualified Teams</div>
                <div class="stat-value">48</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sim Format</div>
                <div class="stat-value">12 Groups</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def coach_note(title, body):
    st.markdown(
        f"""
        <div class="coach-note">
            <strong>{title}</strong><br>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_stat(label, value):
    st.markdown(
        f"""
        <div class="report-stat-card">
            <div class="stat-label">{label}</div>
            <div class="report-stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def soccer_bar_chart(
    df,
    x_column,
    y_column,
    title,
    hint,
    x_axis_title,
    y_axis_title,
    percent=False,
    ascending=False,
    height=330,
):
    if df.empty:
        return
    plot_df = df[[x_column, y_column]].copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column]).sort_values(y_column, ascending=ascending)
    plot_df["value_label"] = plot_df[y_column].map(lambda value: f"{value:.1%}" if percent else f"{value:.3f}")
    value_format = ".1%" if percent else ".2f"
    axis_options = {
        "grid": True,
        "gridColor": "#dce8df",
        "labelColor": "#102017",
        "titleColor": "#0b5d3b",
        "titleFontWeight": 800,
        "titleFontSize": 13,
        "labelFontSize": 12,
    }
    if percent:
        axis_options["format"] = "%"
    max_value = float(plot_df[y_column].max()) if not plot_df.empty else 1
    x_domain_max = 1 if percent else max(max_value * 1.18, max_value + 0.02)
    x_scale = alt.Scale(domain=[0, x_domain_max])
    st.markdown(
        f"""
        <div class="chart-panel">
            <div class="chart-title"><span>{title}</span><span class="chart-badge">Pitch view</span></div>
            <div class="chart-hint">{hint}<br><strong>X-axis:</strong> {x_axis_title}. <strong>Y-axis:</strong> {y_axis_title}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    base = (
        alt.Chart(plot_df)
        .encode(
            y=alt.Y(
                f"{x_column}:N",
                title=y_axis_title,
                sort=list(plot_df[x_column]),
                axis=alt.Axis(
                    labelLimit=260,
                    labelColor="#102017",
                    titleColor="#0b5d3b",
                    titleFontWeight=800,
                    titleFontSize=13,
                    labelFontSize=12,
                ),
            )
        )
    )
    bars = (
        base
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(
                f"{y_column}:Q",
                title=x_axis_title,
                axis=alt.Axis(**axis_options),
                scale=x_scale,
            ),
            color=alt.Color(
                f"{y_column}:Q",
                scale=alt.Scale(range=["#8fd8aa", "#0b6b43"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{x_column}:N", title=x_column),
                alt.Tooltip(f"{y_column}:Q", title=y_column, format=value_format),
            ],
        )
    )
    labels = (
        base
        .mark_text(
            align="left",
            baseline="middle",
            dx=8,
            color="#102017",
            fontSize=12,
            fontWeight=700,
        )
        .encode(
            x=alt.X(f"{y_column}:Q", scale=x_scale),
            text="value_label:N",
        )
    )
    chart = (
        (bars + labels)
        .properties(
            height=height,
            width="container",
            background="#f7fbf6",
            title=alt.TitleParams(
                text=title,
                subtitle=hint,
                anchor="start",
                color="#102017",
                subtitleColor="#496256",
                fontSize=17,
                subtitleFontSize=12,
                dy=-4,
            ),
        )
        .configure(background="#f7fbf6")
        .configure_view(stroke=None, fill="#f7fbf6")
        .configure_axis(
            labelColor="#102017",
            titleColor="#0b5d3b",
            titleFontWeight=800,
            titleFontSize=13,
            labelFontSize=12,
            gridColor="#dce8df",
            domainColor="#496256",
            tickColor="#496256",
        )
        .configure_title(fontWeight=900)
    )
    st.altair_chart(chart, use_container_width=True)


def render_scoreboard(team_a, team_b, winner, prob_a, prob_b):
    st.markdown(
        f"""
        <div class="scoreboard">
            <div class="score-row">
                <div>
                    <div class="team-name">{team_a}</div>
                    <div>{prob_a:.2%}</div>
                </div>
                <div class="vs-badge">VS</div>
                <div>
                    <div class="team-name">{team_b}</div>
                    <div>{prob_b:.2%}</div>
                </div>
            </div>
            <div class="winner-banner">
                <strong>{winner}</strong> is the model favorite for this matchup.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_group_card(group_name, group_teams):
    teams_markup = badge_line(group_teams)
    st.markdown(
        f"""
        <div class="section-card">
            <div class="stat-label">Group {group_name}</div>
            <div style="margin-top:0.45rem;">{teams_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_match(team_a_name, team_b_name, team_features, model):
    team_a = team_row(team_a_name, team_features)
    team_b = team_row(team_b_name, team_features)

    input_data = pd.DataFrame(
        {
            "overall_diff": [team_a["avg_overall"] - team_b["avg_overall"]],
            "attack_diff": [team_a["attack_strength"] - team_b["attack_strength"]],
            "midfield_diff": [team_a["midfield_strength"] - team_b["midfield_strength"]],
            "defense_diff": [team_a["defense_strength"] - team_b["defense_strength"]],
            "goalkeeper_diff": [team_a["goalkeeper_strength"] - team_b["goalkeeper_strength"]],
            "age_diff": [team_a["avg_age"] - team_b["avg_age"]],
            "balance_diff": [team_a["team_balance"] - team_b["team_balance"]],
            "player_pool_diff": [team_a["num_players"] - team_b["num_players"]],
            "elite_player_diff": [team_a["elite_players"] - team_b["elite_players"]],
            "rank_advantage": [team_b["rank"] - team_a["rank"]],
            "points_diff": [team_a["total_points"] - team_b["total_points"]],
            "recent_win_rate_diff_5": [team_a["recent_win_rate_5"] - team_b["recent_win_rate_5"]],
            "recent_goal_diff_diff_10": [team_a["recent_goal_diff_10"] - team_b["recent_goal_diff_10"]],
            "recent_goals_for_diff_10": [team_a["recent_goals_for_10"] - team_b["recent_goals_for_10"]],
            "recent_goals_against_diff_10": [
                team_a["recent_goals_against_10"] - team_b["recent_goals_against_10"]
            ],
            "recent_match_count_diff": [team_a["recent_matches"] - team_b["recent_matches"]],
        }
    )
    input_data = add_engineered_match_features(input_data)[FEATURES]

    prob_a = model.predict_proba(input_data)[0][1]
    prob_b = 1 - prob_a

    return {
        "winner": team_a_name if prob_a > prob_b else team_b_name,
        "team_a_probability": prob_a,
        "team_b_probability": prob_b,
        "input_data": input_data,
    }


@st.cache_data
def load_world_cup_groups():
    groups_path = REFERENCE_DATA_DIR / "2026_world_cup_groups.csv"
    groups = pd.read_csv(groups_path)
    groups.columns = groups.columns.str.strip().str.lower().str.replace(" ", "_")
    groups["team"] = groups["team"].replace(TEAM_NAME_ALIASES)
    return groups


def build_probability_lookup(groups, team_features, model):
    teams_in_groups = groups["team"].dropna().unique()
    ready_teams = set(prediction_ready_teams(team_features))
    teams_in_model = [team for team in teams_in_groups if team in ready_teams]

    probabilities = {}
    for team_a in teams_in_model:
        for team_b in teams_in_model:
            if team_a == team_b:
                continue
            result = predict_match(team_a, team_b, team_features, model)
            probabilities[(team_a, team_b)] = result["team_a_probability"]

    return probabilities, teams_in_model


def simulate_game(team_a, team_b, probabilities, rng, allow_draw=False):
    prob_a = probabilities[(team_a, team_b)]

    if allow_draw:
        closeness = 1 - min(abs(prob_a - 0.5) * 2, 1)
        draw_probability = 0.18 + (0.12 * closeness)
        win_probability = (1 - draw_probability) * prob_a
        roll = rng.random()

        if roll < win_probability:
            return team_a
        if roll < win_probability + draw_probability:
            return "Draw"
        return team_b

    return team_a if rng.random() < prob_a else team_b


def simulate_group(group_name, group_teams, probabilities, rng):
    table = {
        team: {
            "group": group_name,
            "team": team,
            "group_position": 0,
            "points": 0,
            "wins": 0,
            "goal_diff": 0,
            "goals_for": 0,
        }
        for team in group_teams
    }

    for i, team_a in enumerate(group_teams):
        for team_b in group_teams[i + 1 :]:
            result = simulate_game(team_a, team_b, probabilities, rng, allow_draw=True)
            strength_gap = abs(probabilities[(team_a, team_b)] - 0.5)
            goal_margin = int(rng.choice([1, 1, 2, 2, 3], p=[0.34, 0.26, 0.22, 0.13, 0.05]))
            if strength_gap > 0.25:
                goal_margin += int(rng.random() < 0.25)

            if result == "Draw":
                draw_goals = int(rng.choice([0, 1, 1, 2], p=[0.25, 0.45, 0.2, 0.1]))
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1
                table[team_a]["goals_for"] += draw_goals
                table[team_b]["goals_for"] += draw_goals
            elif result == team_a:
                loser_goals = int(rng.choice([0, 0, 1, 1, 2], p=[0.34, 0.22, 0.25, 0.14, 0.05]))
                table[team_a]["points"] += 3
                table[team_a]["wins"] += 1
                table[team_a]["goal_diff"] += goal_margin
                table[team_a]["goals_for"] += loser_goals + goal_margin
                table[team_b]["goal_diff"] -= goal_margin
                table[team_b]["goals_for"] += loser_goals
            else:
                loser_goals = int(rng.choice([0, 0, 1, 1, 2], p=[0.34, 0.22, 0.25, 0.14, 0.05]))
                table[team_b]["points"] += 3
                table[team_b]["wins"] += 1
                table[team_b]["goal_diff"] += goal_margin
                table[team_b]["goals_for"] += loser_goals + goal_margin
                table[team_a]["goal_diff"] -= goal_margin
                table[team_a]["goals_for"] += loser_goals

    ranked = sorted(
        table.values(),
        key=lambda row: (
            row["points"],
            row["goal_diff"],
            row["goals_for"],
            row["wins"],
            rng.random(),
        ),
        reverse=True,
    )

    for position, row in enumerate(ranked, start=1):
        row["group_position"] = position

    return ranked


def rank_qualified_teams(group_tables, rng):
    direct_qualifiers = []
    third_place_teams = []
    for table in group_tables.values():
        direct_qualifiers.extend(table[:2])
        third_place_teams.append(table[2])

    best_thirds = sorted(
        third_place_teams,
        key=lambda row: (
            row["points"],
            row["goal_diff"],
            row["goals_for"],
            row["wins"],
            rng.random(),
        ),
        reverse=True,
    )[:8]

    qualifiers = direct_qualifiers + best_thirds
    seeded = sorted(
        qualifiers,
        key=lambda row: (
            row["group_position"] == 1,
            row["group_position"] == 2,
            row["points"],
            row["goal_diff"],
            row["goals_for"],
            row["wins"],
            rng.random(),
        ),
        reverse=True,
    )

    return seeded


def build_seeded_knockout_pairings(qualified_teams):
    teams = [row["team"] for row in qualified_teams]
    bracket_order = [0, 31, 15, 16, 7, 24, 8, 23, 3, 28, 12, 19, 4, 27, 11, 20,
                     1, 30, 14, 17, 6, 25, 9, 22, 2, 29, 13, 18, 5, 26, 10, 21]
    ordered = [teams[index] for index in bracket_order]
    return list(zip(ordered[0::2], ordered[1::2]))


def play_knockout_round(pairings, probabilities, rng):
    return [simulate_game(team_a, team_b, probabilities, rng) for team_a, team_b in pairings]


def pair_adjacent(teams):
    return list(zip(teams[0::2], teams[1::2]))


def simulate_tournament(groups, probabilities, rng):
    group_tables = {}
    for group_name, group_df in groups.groupby("group", sort=True):
        teams = group_df["team"].tolist()
        group_tables[group_name] = simulate_group(group_name, teams, probabilities, rng)

    qualified_teams = rank_qualified_teams(group_tables, rng)
    round_of_32_pairings = build_seeded_knockout_pairings(qualified_teams)
    round_of_16_teams = play_knockout_round(round_of_32_pairings, probabilities, rng)
    quarterfinalists = play_knockout_round(pair_adjacent(round_of_16_teams), probabilities, rng)
    semifinalists = play_knockout_round(pair_adjacent(quarterfinalists), probabilities, rng)
    finalists = play_knockout_round(pair_adjacent(semifinalists), probabilities, rng)
    champion = play_knockout_round(pair_adjacent(finalists), probabilities, rng)[0]

    return {
        "group_winners": [table[0]["team"] for table in group_tables.values()],
        "round_of_32": [row["team"] for row in qualified_teams],
        "round_of_16": round_of_16_teams,
        "quarterfinals": quarterfinalists,
        "semifinals": semifinalists,
        "finalists": finalists,
        "champion": champion,
    }


def run_world_cup_simulations(groups, team_features, model, simulation_count, seed):
    probabilities, teams_in_model = build_probability_lookup(groups, team_features, model)
    missing_teams = sorted(set(groups["team"]) - set(teams_in_model))

    if missing_teams:
        return None, missing_teams, None

    rng = np.random.default_rng(seed)
    counts = {
        team: {
            "Group Winner": 0,
            "Round of 32": 0,
            "Round of 16": 0,
            "Quarterfinalist": 0,
            "Semifinalist": 0,
            "Finalist": 0,
            "Champion": 0,
        }
        for team in groups["team"].unique()
    }

    sample_result = None
    for _ in range(simulation_count):
        result = simulate_tournament(groups, probabilities, rng)
        if sample_result is None:
            sample_result = result
        for team in result["group_winners"]:
            counts[team]["Group Winner"] += 1
        for team in result["round_of_32"]:
            counts[team]["Round of 32"] += 1
        for team in result["round_of_16"]:
            counts[team]["Round of 16"] += 1
        for team in result["quarterfinals"]:
            counts[team]["Quarterfinalist"] += 1
        for team in result["semifinals"]:
            counts[team]["Semifinalist"] += 1
        for team in result["finalists"]:
            counts[team]["Finalist"] += 1
        counts[result["champion"]]["Champion"] += 1

    summary = pd.DataFrame.from_dict(counts, orient="index").reset_index(names="Team")
    for column in summary.columns[1:]:
        summary[column] = summary[column] / simulation_count

    return summary.sort_values("Champion", ascending=False), missing_teams, sample_result


def render_simulation_result(summary, sample_result, model_name, model_accuracy=None):
    champion = summary.iloc[0]
    st.markdown(
        f"""
        <div class="winner-banner">
            <div class="stat-label" style="color:rgba(255,255,255,0.78);">Most likely champion with {model_name}</div>
            <div style="font-size:2rem;font-weight:900;">{champion['Team']}</div>
            <div>Champion rate: <strong>{champion['Champion']:.2%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if model_accuracy is not None:
        st.caption(f"Model test accuracy: {model_accuracy:.2%}. Knockout rounds use a seeded 32-team bracket based on simulated group performance.")
    else:
        st.caption("Knockout rounds use a seeded 32-team bracket based on simulated group performance.")
    st.info(champion_explanation(champion["Team"], team_features))
    st.markdown(
        f'<div class="badge-row">Top champion badge: {team_badge(champion["Team"])}</div>',
        unsafe_allow_html=True,
    )

    top_champions = summary[["Team", "Champion"]].head(10)
    top_finalists = summary[["Team", "Finalist"]].sort_values("Finalist", ascending=False).head(10)
    top_semifinalists = summary[["Team", "Semifinalist"]].sort_values("Semifinalist", ascending=False).head(10)

    st.subheader("Most Likely Champions")
    st.markdown(
        f'<div class="badge-row">{badge_line(top_champions["Team"].head(5).tolist())}</div>',
        unsafe_allow_html=True,
    )
    soccer_bar_chart(
        top_champions,
        "Team",
        "Champion",
        f"Most Likely 2026 World Cup Champions with {model_name}",
        "Each bar shows the percentage of simulated tournaments won by that team. Longer bars mean the team survives the group stage and knockout bracket more often.",
        "Champion probability across simulations",
        "National team",
        percent=True,
        height=360,
    )
    st.caption(
        f"Takeaway: {top_champions.iloc[0]['Team']} has the strongest title path for {model_name}, "
        f"winning {top_champions.iloc[0]['Champion']:.1%} of tournaments."
    )

    st.subheader("Sample Knockout Bracket")
    st.caption("One simulated tournament path from this model's run, shown as a bracket-style snapshot.")
    render_bracket(sample_result)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.subheader("Most Likely Finalists")
        st.dataframe(format_percent_table(top_finalists), use_container_width=True, hide_index=True)
    with sim_col2:
        st.subheader("Most Likely Semifinalists")
        st.dataframe(format_percent_table(top_semifinalists), use_container_width=True, hide_index=True)

    st.subheader("Full Simulation Table")
    with st.expander(f"View full simulation table for {model_name}"):
        st.dataframe(format_percent_table(summary), use_container_width=True, hide_index=True)
    st.download_button(
        f"Download {model_name} simulation results",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"2026_world_cup_{model_name}_simulation_results.csv".replace(" ", "_"),
        mime="text/csv",
        use_container_width=True,
    )


st.set_page_config(page_title="2026 FIFA Predictor", page_icon="soccer", layout="wide")

try:
    model, match_models, team_features, metrics = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

theme_mode = st.sidebar.radio(
    "App theme",
    ["Stadium Mode", "Clean Report Mode"],
    horizontal=True,
)
inject_soccer_theme(theme_mode)
render_hero(metrics)

with st.expander("Demo Script"):
    st.write("- Predict a match: load Brazil vs France, run the prediction, and explain the probability gap.")
    st.write("- Simulate the 2026 World Cup: load demo settings, run 1,000 simulations, and compare champion/finalist rates.")
    st.write("- Explain the model: show best-model accuracy, confusion matrix, feature importance, and known data gaps.")

match_tab, simulator_tab, game_tab, report_tab = st.tabs(
    ["Predict a Match", "Simulate 2026 World Cup", "Play Penalty Shootout", "How the Model Works"]
)

with match_tab:
    only_qualified = st.toggle("2026 qualified teams only", value=True)
    teams = qualified_prediction_teams(team_features) if only_qualified else prediction_ready_teams(team_features)
    missing_qualified = sorted(set(QUALIFIED_2026_TEAMS) - set(qualified_prediction_teams(team_features)))
    coach_note(
        "Scout report",
        "Use the team comparison like a pre-match sheet: ranking points capture long-term quality, recent form captures momentum, and player-strength gaps show where the matchup may tilt.",
    )

    if missing_qualified:
        st.warning("Missing qualified teams: " + ", ".join(missing_qualified))
    elif only_qualified:
        st.success("All 48 qualified 2026 World Cup teams are available.")

    demo_col1, demo_col2 = st.columns([1, 2])
    with demo_col1:
        if st.button("Load Demo Match", use_container_width=True):
            st.session_state["team_a_select"] = "Brazil" if "Brazil" in teams else teams[0]
            st.session_state["team_b_select"] = "France" if "France" in teams else teams[min(1, len(teams) - 1)]
            st.session_state["demo_match_loaded"] = True
    with demo_col2:
        if st.session_state.get("demo_match_loaded"):
            st.info("Demo match loaded. Click Predict to show the model explanation.")

    col1, col2 = st.columns(2)
    default_a = teams.index("Brazil") if "Brazil" in teams else 0
    default_b = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
    with col1:
        team_a = st.selectbox("Team A", teams, index=default_a, key="team_a_select")
    with col2:
        team_b = st.selectbox("Team B", teams, index=default_b, key="team_b_select")

    st.markdown(
        f'<div class="badge-row">{team_badge(team_a)} <span class="vs-badge">VS</span> {team_badge(team_b)}</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Team Comparison")
    comparison_df = team_comparison_table(team_a, team_b, team_features)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if st.button("Predict", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.warning("Pick two different teams.")
        else:
            result = predict_match(team_a, team_b, team_features, model)
            winner = result["winner"]
            prob_a = result["team_a_probability"]
            prob_b = result["team_b_probability"]
            confidence, probability_gap = confidence_label(prob_a, prob_b)
            explanation = prediction_explanation(result, result["input_data"], team_a, team_b)
            if winner in COUNTRY_THEMES:
                st.session_state["penalty_country_theme"] = winner

            render_scoreboard(team_a, team_b, winner, prob_a, prob_b)
            metric_a, metric_b, metric_c = st.columns(3)
            with metric_a:
                st.metric(team_a, f"{prob_a:.2%}")
            with metric_b:
                st.metric(team_b, f"{prob_b:.2%}")
            with metric_c:
                st.metric("Confidence", confidence, f"{probability_gap:.1%} gap")

            st.progress(float(prob_a), text=f"{team_a} win probability")
            st.info(explanation)
            st.success(f"Prediction favors {winner}")

            prediction_export = comparison_df.copy()
            prediction_export["Predicted winner"] = winner
            prediction_export[f"{team_a} win probability"] = prob_a
            prediction_export[f"{team_b} win probability"] = prob_b
            prediction_export["Confidence"] = confidence
            prediction_export["Explanation"] = explanation
            st.download_button(
                "Download match comparison",
                data=prediction_export.to_csv(index=False).encode("utf-8"),
                file_name=f"{team_a}_vs_{team_b}_prediction.csv".replace(" ", "_"),
                mime="text/csv",
                use_container_width=True,
            )

with simulator_tab:
    groups = load_world_cup_groups()
    st.caption("2026 format: 12 groups, top two from each group plus the eight best third-place teams advance.")
    coach_note(
        "Tournament hint",
        "Run at least 1,000 simulations for a steadier table. A single bracket can swing on one upset, but repeated runs show which teams keep surviving the tournament path.",
    )

    group_grid = groups.groupby("group")["team"].apply(list)
    with st.expander("View 2026 groups"):
        group_cols = st.columns(4)
        for index, (group_name, group_teams) in enumerate(group_grid.items()):
            with group_cols[index % 4]:
                render_group_card(group_name, group_teams)

    demo_sim_col1, demo_sim_col2 = st.columns([1, 2])
    with demo_sim_col1:
        if st.button("Load Demo Settings", use_container_width=True):
            st.session_state["simulation_count_slider"] = 1000
            st.session_state["simulation_seed_input"] = 42
            st.session_state["demo_sim_loaded"] = True
    with demo_sim_col2:
        if st.session_state.get("demo_sim_loaded"):
            st.info("Demo settings loaded. Click Run World Cup Simulation for a stable walkthrough.")

    simulation_count = st.slider(
        "Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="simulation_count_slider",
    )
    seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
        key="simulation_seed_input",
    )

    if st.button("Run World Cup Simulation", use_container_width=True, type="primary"):
        simulation_results = {}
        simulation_missing = []
        available_simulation_models = {
            name: match_models[name]
            for name in SIMULATION_MODEL_NAMES
            if name in match_models
        }
        if not available_simulation_models:
            available_simulation_models = {metrics.get("best_match_model", "Best Model"): model}

        with st.spinner(f"Running {simulation_count:,} tournament simulations for {len(available_simulation_models)} models..."):
            for model_index, (model_name, candidate_model) in enumerate(available_simulation_models.items()):
                summary, missing_teams, sample_result = run_world_cup_simulations(
                    groups,
                    team_features,
                    candidate_model,
                    simulation_count,
                    int(seed) + model_index,
                )
                if missing_teams:
                    simulation_missing = missing_teams
                    continue
                simulation_results[model_name] = {
                    "summary": summary,
                    "sample_result": sample_result,
                }

        st.session_state["simulation_results_by_model"] = simulation_results
        st.session_state["simulation_missing_teams"] = simulation_missing
        st.session_state["simulation_count"] = simulation_count
        st.session_state["simulation_seed"] = int(seed)

    simulation_results = st.session_state.get("simulation_results_by_model", {})
    missing_teams = st.session_state.get("simulation_missing_teams", [])

    if simulation_results or missing_teams:
        if missing_teams:
            st.error(
                "Some group teams are missing model-ready features: "
                + ", ".join(missing_teams)
            )
        else:
            best_model_name = metrics.get("best_match_model")
            tab_labels = [
                f"{model_name} 👑" if model_name == best_model_name else model_name
                for model_name in simulation_results
            ]
            model_tabs = st.tabs(tab_labels)
            for tab, (model_name, result) in zip(model_tabs, simulation_results.items()):
                with tab:
                    model_accuracy = (
                        metrics.get("classification_models", {})
                        .get(model_name, {})
                        .get("accuracy")
                    )
                    render_simulation_result(
                        result["summary"],
                        result["sample_result"],
                        model_name,
                        model_accuracy,
                    )

with game_tab:
    render_penalty_game()

with report_tab:
    present_qualified = qualified_prediction_teams(team_features)
    missing_qualified = sorted(set(QUALIFIED_2026_TEAMS) - set(present_qualified))

    c1, c2, c3 = st.columns(3)
    with c1:
        report_stat("Best match model", metrics.get("best_match_model", "N/A"))
    with c2:
        report_stat("Match accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with c3:
        report_stat("2026 teams available", f"{len(present_qualified)}/48")

    c4, c5 = st.columns(2)
    with c4:
        report_stat("Training matches", f"{metrics.get('training_rows', 0):,}")
    with c5:
        report_stat("Tournament-success rows", f"{metrics.get('tournament_success_rows', 0):,}")

    if missing_qualified:
        st.warning("Missing teams: " + ", ".join(missing_qualified))
    else:
        st.success("All 48 qualified 2026 World Cup teams are available in the predictor.")

    coach_note(
        "Model reading guide",
        "Treat all-match accuracy like the full league table and high-confidence accuracy like clear scoring chances. The model is stronger when the matchup has obvious signal.",
    )

    st.subheader("Pipeline")
    pipeline_steps = [
        "Loaded and cleaned players, matches, rankings, World Cup history, and country-name datasets.",
        "Aligned country names across datasets with explicit aliases.",
        "Created goal difference and home-win target variables.",
        "Built team features from player data: overall, attack, midfield, defense, goalkeeper, age, balance, elite players, and player pool size.",
        "Merged latest FIFA rankings into team profiles and used time-aware ranking snapshots for historical match training.",
        "Added rolling recent-form features before each match: win rate, goal difference, scoring, defending, and match sample size.",
        "Converted team data into match-level feature differences.",
        "Filtered to matches from 2005 onward for modern-football training data.",
        "Compared Logistic Regression, Polynomial Logistic Regression, Random Forest, tuned Random Forest, SVM, Naive Bayes, MLP, and gradient boosting models for match-outcome prediction.",
        "Compared Linear, Ridge, Lasso, Random Forest, SVR, and MLP regressors for goal-difference prediction.",
        "Built a tournament-success dataset to predict advancement past the group stage.",
        "Generated local model artifacts used by the Streamlit app.",
        "Simulated 2026 group and knockout outcomes with repeated Monte Carlo runs.",
    ]
    with st.expander("View modeling pipeline steps"):
        for step in pipeline_steps:
            st.write(f"- {step}")

    st.subheader("Match Outcome Models")
    st.caption(
        "This table compares classifiers on the same held-out test split. "
        "Higher accuracy, precision, recall, F1, and ROC-AUC generally indicate stronger match-outcome performance. "
        "High-confidence accuracy measures only predictions where the model is at least 70% confident."
    )
    classification_df = metrics_table(
        metrics.get("classification_models", {}),
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "high_confidence_accuracy_70",
            "high_confidence_coverage_70",
        ],
    )
    if not classification_df.empty:
        soccer_bar_chart(
            classification_df,
            "Model",
            "ACCURACY",
            "Match Outcome Model Accuracy on Held-Out International Matches",
            "This compares every classifier on the same test split. The X-axis is prediction accuracy, and the Y-axis lists the model families used in the tournament predictor.",
            "Accuracy on held-out matches",
            "Model type",
            percent=True,
            height=390,
        )
        best_row = classification_df.sort_values("ACCURACY", ascending=False).iloc[0]
        st.success(
            f"{best_row['Model']} is the strongest match-outcome model in this run, "
            f"with {best_row['ACCURACY']:.2%} accuracy."
        )
        with st.expander("View classifier metrics table"):
            st.dataframe(
                classification_df.sort_values("ACCURACY", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        high_confidence_accuracy = best_row.get("HIGH_CONFIDENCE_ACCURACY_70")
        high_confidence_coverage = best_row.get("HIGH_CONFIDENCE_COVERAGE_70")
        if pd.notna(high_confidence_accuracy) and pd.notna(high_confidence_coverage):
            st.subheader("How to Read the Accuracy")
            overall_col, confident_col = st.columns(2)
            with overall_col:
                st.metric("All-match accuracy", f"{best_row['ACCURACY']:.1%}")
                st.caption(
                    "This is the fairest headline number because the model must make a prediction "
                    "for every held-out match, including close games, friendlies, and noisy results."
                )
            with confident_col:
                st.metric("High-confidence accuracy", f"{high_confidence_accuracy:.1%}")
                st.caption(
                    f"This is the practical 80%+ number. It only counts matches where the model is "
                    f"at least 70% confident, covering {high_confidence_coverage:.1%} of held-out matches."
                )
            st.write(
                "Both numbers are useful: the all-match score shows how the model performs without filtering, "
                "while the high-confidence score shows how accurate it can be when it has a clear signal from "
                "rankings, recent form, and team-strength differences."
            )

    matrix_df = confusion_matrix_table(metrics.get("confusion_matrix", []))
    if not matrix_df.empty:
        st.subheader("Best Model Confusion Matrix")
        st.caption("Rows are actual outcomes and columns are model predictions for the best match-outcome classifier.")
        with st.expander("View confusion matrix"):
            st.dataframe(matrix_df, use_container_width=True)

    st.subheader("Goal Difference Models")
    st.caption(
        "Regression models estimate expected goal difference. Lower MAE and RMSE are better; higher R-squared is better."
    )
    regression_df = metrics_table(
        metrics.get("regression_models", {}),
        ["mae", "rmse", "r2"],
    )
    if not regression_df.empty:
        soccer_bar_chart(
            regression_df,
            "Model",
            "RMSE",
            "Goal Difference Regression Error by Model",
            "RMSE measures how far each model's predicted goal margin is from the actual goal margin. Lower values mean the score-margin estimate is more stable.",
            "RMSE in goals",
            "Regression model",
            ascending=True,
            height=340,
        )
        best_regression = regression_df.sort_values("RMSE", ascending=True).iloc[0]
        st.success(
            f"{best_regression['Model']} has the lowest RMSE in this run, "
            f"so it is the strongest goal-difference estimator among the tested models."
        )
        with st.expander("View goal-difference metrics table"):
            st.dataframe(
                regression_df.sort_values("RMSE", ascending=True),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Tournament Success Models")
    st.caption(
        "These models estimate whether a team advances past the group stage using team-level features."
    )
    tournament_df = metrics_table(
        metrics.get("tournament_success_models", {}),
        ["accuracy", "roc_auc", "precision", "recall", "f1"],
    )
    if not tournament_df.empty:
        soccer_bar_chart(
            tournament_df,
            "Model",
            "ACCURACY",
            "Tournament Group Advancement Accuracy by Model",
            "This chart compares models for predicting whether a team advances past the World Cup group stage. The dataset is smaller, so treat this as supporting evidence.",
            "Accuracy predicting group advancement",
            "Tournament-success model",
            percent=True,
            height=310,
        )
        best_tournament = tournament_df.sort_values("ACCURACY", ascending=False).iloc[0]
        st.warning(
            f"{best_tournament['Model']} has the highest tournament-success accuracy, "
            "but this task uses fewer World Cup rows, so results should be interpreted cautiously."
        )
        with st.expander("View tournament-success metrics table"):
            st.dataframe(
                tournament_df.sort_values("ACCURACY", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    feature_importance = pd.DataFrame(metrics.get("feature_importance", []))
    if not feature_importance.empty:
        importance_model = metrics.get("feature_importance_model", "available model")
        st.subheader(f"Feature Importance ({importance_model})")
        st.caption(
            "Feature importance comes from the Random Forest model because the best polynomial model is not as directly interpretable."
        )
        soccer_bar_chart(
            feature_importance.head(12),
            "feature",
            "importance",
            "Top Prediction Drivers in the Match Outcome Model",
            "Feature importance shows which inputs the model relies on most. Ranking points, recent form, and team-strength differences act like the model's scouting report.",
            "Relative feature importance",
            "Model feature",
            height=420,
        )
        st.caption(
            "Takeaway: ranking signal, recent form, and squad-strength differences are the model's clearest prediction drivers."
        )
        with st.expander("View full feature-importance table"):
            st.dataframe(feature_importance, use_container_width=True, hide_index=True)

    data_gaps = metrics.get("data_gaps", [])
    if data_gaps:
        st.subheader("Known Data Gaps")
        with st.expander("View data limitations"):
            for gap in data_gaps:
                st.write(f"- {gap}")

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        add_file_download(
            "Download model dataset",
            PROCESSED_DATA_DIR / "compiled_fifa_match_dataset.csv",
            "text/csv",
        )
    with d2:
        add_file_download(
            "Download capstone report",
            DOCS_DIR / "CAPSTONE_REPORT.md",
            "text/markdown",
        )
    with d3:
        add_file_download(
            "Download presentation deck",
            DOCS_DIR / "FIFA_World_Cup_Predictor_Capstone.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

st.divider()
st.caption(
    f"Model: {metrics.get('best_match_model', 'trained classifier')} | "
    "Data: international matches from 2005 onward | "
    "Known limitation: possession, shots, shots on target, fouls, and the exact 2018 squad roster file were not available locally."
)
