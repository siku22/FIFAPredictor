from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REFERENCE_DATA_DIR = ROOT_DIR / "data" / "reference"
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


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "fifa_model.pkl"
    features_path = MODELS_DIR / "team_features.pkl"
    metrics_path = MODELS_DIR / "model_metrics.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python3 src/train_model.py` first."
        )

    metrics = joblib.load(metrics_path) if metrics_path.exists() else {}
    return joblib.load(model_path), joblib.load(features_path), metrics


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


def inject_soccer_theme():
    st.markdown(
        """
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
                linear-gradient(90deg, rgba(255,255,255,0.055) 1px, transparent 1px),
                linear-gradient(0deg, rgba(255,255,255,0.05) 1px, transparent 1px),
                radial-gradient(circle at 50% 16%, rgba(214,179,90,0.18), transparent 24%),
                linear-gradient(135deg, #08371f 0%, #0b5d3b 48%, #062f22 100%);
            background-size: 72px 72px, 72px 72px, auto, auto;
            color: #f7fbf6;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stTabs"] button {
            color: #ecfff4;
            border-radius: 999px;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            background: rgba(255,255,255,0.16);
            color: #ffffff;
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
            width: 170px;
            height: 170px;
            border: 2px solid rgba(255,255,255,0.18);
            border-radius: 50%;
            top: 50%;
            right: 8%;
            transform: translateY(-50%);
            pointer-events: none;
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
        .section-card {
            background: rgba(255,255,255,0.94);
            color: var(--ink);
            border: 1px solid rgba(255,255,255,0.52);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 18px 40px rgba(0,0,0,0.15);
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

        .scoreboard {
            text-align: center;
            margin-top: 1rem;
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

        @media (max-width: 780px) {
            .stat-strip,
            .score-row {
                grid-template-columns: 1fr;
            }
            .hero-panel {
                padding: 1.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metrics):
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="eyebrow">World Cup analytics dashboard</div>
            <h1 class="hero-title">2026 FIFA World Cup Predictor</h1>
            <div class="hero-copy">
                Match probabilities and tournament simulations powered by player ratings,
                team composition, FIFA rankings, and historical international results.
            </div>
        </div>
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
    teams_markup = "".join(f'<span class="group-pill">{team}</span>' for team in group_teams)
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
    tf = team_features.copy()
    tf["team_clean"] = tf["team"].str.lower().str.strip()

    team_a = tf[tf["team_clean"] == team_a_name.lower().strip()].iloc[0]
    team_b = tf[tf["team_clean"] == team_b_name.lower().strip()].iloc[0]

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
        }
    )[FEATURES]

    prob_a = model.predict_proba(input_data)[0][1]
    prob_b = 1 - prob_a

    return {
        "winner": team_a_name if prob_a > prob_b else team_b_name,
        "team_a_probability": prob_a,
        "team_b_probability": prob_b,
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
        return None, missing_teams

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

    for _ in range(simulation_count):
        result = simulate_tournament(groups, probabilities, rng)
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

    return summary.sort_values("Champion", ascending=False), missing_teams


st.set_page_config(page_title="2026 FIFA Predictor", page_icon="soccer", layout="wide")

try:
    model, team_features, metrics = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

inject_soccer_theme()
render_hero(metrics)

match_tab, simulator_tab, report_tab = st.tabs(["Match Predictor", "World Cup Simulator", "Model Report"])

with match_tab:
    only_qualified = st.toggle("2026 qualified teams only", value=True)
    teams = qualified_prediction_teams(team_features) if only_qualified else prediction_ready_teams(team_features)
    missing_qualified = sorted(set(QUALIFIED_2026_TEAMS) - set(qualified_prediction_teams(team_features)))

    if missing_qualified:
        st.warning("Missing qualified teams: " + ", ".join(missing_qualified))
    elif only_qualified:
        st.success("All 48 qualified 2026 World Cup teams are available.")

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

            render_scoreboard(team_a, team_b, winner, prob_a, prob_b)
            metric_a, metric_b = st.columns(2)
            with metric_a:
                st.metric(team_a, f"{prob_a:.2%}")
            with metric_b:
                st.metric(team_b, f"{prob_b:.2%}")

            st.progress(float(prob_a), text=f"{team_a} win probability")
            st.success(f"Prediction favors {winner}")

with simulator_tab:
    groups = load_world_cup_groups()
    st.caption("2026 format: 12 groups, top two from each group plus the eight best third-place teams advance.")

    group_grid = groups.groupby("group")["team"].apply(list)
    with st.expander("View 2026 groups"):
        group_cols = st.columns(4)
        for index, (group_name, group_teams) in enumerate(group_grid.items()):
            with group_cols[index % 4]:
                render_group_card(group_name, group_teams)

    simulation_count = st.slider(
        "Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    if st.button("Run World Cup Simulation", use_container_width=True):
        with st.spinner(f"Running {simulation_count:,} tournament simulations..."):
            summary, missing_teams = run_world_cup_simulations(
                groups, team_features, model, simulation_count, int(seed)
            )

        if missing_teams:
            st.error(
                "Some group teams are missing model-ready features: "
                + ", ".join(missing_teams)
            )
        else:
            champion = summary.iloc[0]
            st.markdown(
                f"""
                <div class="winner-banner">
                    <div class="stat-label" style="color:rgba(255,255,255,0.78);">Most likely champion</div>
                    <div style="font-size:2rem;font-weight:900;">{champion['Team']}</div>
                    <div>Champion rate: <strong>{champion['Champion']:.2%}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Knockout rounds use a seeded 32-team bracket based on simulated group performance.")

            chart_data = summary.set_index("Team")["Champion"].head(12)
            st.bar_chart(chart_data)

            display_summary = summary.copy()
            for column in display_summary.columns[1:]:
                display_summary[column] = display_summary[column].map("{:.2%}".format)

            st.dataframe(display_summary, use_container_width=True, hide_index=True)

with report_tab:
    present_qualified = qualified_prediction_teams(team_features)
    missing_qualified = sorted(set(QUALIFIED_2026_TEAMS) - set(present_qualified))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best match model", metrics.get("best_match_model", "N/A"))
    with c2:
        st.metric("Match accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with c3:
        st.metric("2026 teams available", f"{len(present_qualified)}/48")

    c4, c5 = st.columns(2)
    with c4:
        st.metric("Training matches", f"{metrics.get('training_rows', 0):,}")
    with c5:
        st.metric("Tournament-success rows", f"{metrics.get('tournament_success_rows', 0):,}")

    if missing_qualified:
        st.warning("Missing teams: " + ", ".join(missing_qualified))
    else:
        st.success("All 48 qualified 2026 World Cup teams are available in the predictor.")

    st.subheader("Pipeline")
    pipeline_steps = [
        "Loaded and cleaned players, matches, rankings, squads, and country-name datasets.",
        "Aligned country names across datasets with explicit aliases.",
        "Created goal difference and home-win target variables.",
        "Built team features from player data: overall, attack, midfield, defense, goalkeeper, age, balance, elite players, and player pool size.",
        "Merged FIFA rankings and points into team-level features.",
        "Converted team data into match-level feature differences.",
        "Filtered to matches from 2005 onward for modern-football training data.",
        "Compared Logistic Regression, Random Forest, SVM, Naive Bayes, and MLP for match-outcome prediction.",
        "Compared Linear, Ridge, Lasso, Random Forest, SVR, and MLP regressors for goal-difference prediction.",
        "Built a tournament-success dataset to predict advancement past the group stage.",
        "Generated local model artifacts used by the Streamlit app.",
        "Simulated 2026 group and knockout outcomes with repeated Monte Carlo runs.",
    ]
    for step in pipeline_steps:
        st.write(f"- {step}")

    st.subheader("Match Outcome Models")
    classification_df = metrics_table(
        metrics.get("classification_models", {}),
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    if not classification_df.empty:
        st.dataframe(
            classification_df.sort_values("ACCURACY", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Goal Difference Models")
    regression_df = metrics_table(
        metrics.get("regression_models", {}),
        ["mae", "rmse", "r2"],
    )
    if not regression_df.empty:
        st.dataframe(
            regression_df.sort_values("RMSE", ascending=True),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Tournament Success Models")
    tournament_df = metrics_table(
        metrics.get("tournament_success_models", {}),
        ["accuracy", "roc_auc", "precision", "recall", "f1"],
    )
    if not tournament_df.empty:
        st.dataframe(
            tournament_df.sort_values("ACCURACY", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    feature_importance = pd.DataFrame(metrics.get("feature_importance", []))
    if not feature_importance.empty:
        st.subheader("Feature Importance")
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)

    data_gaps = metrics.get("data_gaps", [])
    if data_gaps:
        st.subheader("Known Data Gaps")
        for gap in data_gaps:
            st.write(f"- {gap}")

st.divider()
st.caption(f"Model: {metrics.get('best_match_model', 'trained classifier')} | Data: international matches from 2005 onward")
