# Predicting FIFA World Cup Outcomes Using Player Attributes, Team Composition, and Match-Level Analysis

## Project Overview

Predicting outcomes in international football tournaments is difficult because match results depend on player ability, squad structure, historical performance, rankings, and tournament format. This project builds a structured predictive modeling framework that connects player-level FIFA attributes to team-level features, converts those features into match-level comparisons, and uses them to predict match outcomes, goal difference, and tournament success.

The project also extends the validated match-outcome model into a forward-looking 2026 FIFA World Cup simulator. Instead of producing one deterministic bracket, the simulator runs repeated tournament simulations and reports probabilistic advancement and championship rates.

## Data Sources

The local project uses these datasets:

- `data/raw/fifa18_clean.csv`: FIFA player attributes, including overall rating, age, position, and nationality.
- `data/raw/international_matches.csv`: historical international match results.
- `data/raw/fifa_mens_rank.csv`: FIFA ranking and points data.
- `data/raw/former_names.csv`: country-name mapping used to align teams across datasets.
- `data/raw/world_cup_matches.csv`: historical World Cup match and stage data.
- `data/reference/2026_world_cup_groups.csv`: hand-curated 2026 group assignments used by the simulator.

The requested `world_cup_2018_squads.xlsx` and additional match-stat file containing possession, shots, shots on target, and fouls are not present in the local project folder. Because of that, the current implementation derives team strength from FIFA18 nationality-level player records rather than exact 2018 roster membership, and it does not include possession, shot, shot-on-target, or foul differences.

## Data Integration Pipeline

The project follows this structure:

```text
raw datasets
-> cleaned team names
-> player-derived team-level features
-> match-level feature differences
-> match outcome, goal difference, and tournament success models
-> 2026 World Cup simulation
-> Streamlit dashboard
```

Country names are standardized with explicit mappings so records such as `USA`, `IR Iran`, `Korea Republic`, `Turkey`, and `Côte d'Ivoire` align with the names used in ranking, match, and 2026 group data.

## Variables

### Predictive Targets

The project evaluates three modeling tasks:

- Match outcome classification: whether the home or first-listed team wins.
- Goal difference regression: home goals minus away goals.
- Tournament success classification: whether a team advances past the group stage.

### Team-Level Explanatory Variables

Team features are derived from FIFA player data and ranking data:

- Average overall rating
- Average attacking strength
- Average midfield strength
- Average defensive strength
- Goalkeeper strength
- Average age
- Team balance, measured as the standard deviation of overall ratings
- Player pool size
- Elite-player count, defined as players rated 80 or higher
- FIFA rank
- FIFA total points

### Match-Level Explanatory Variables

Match features are constructed as differences between Team A and Team B:

- Overall rating difference
- Attack difference
- Midfield difference
- Defense difference
- Goalkeeper difference
- Age difference
- Balance difference
- Player-pool difference
- Elite-player difference
- Rank advantage
- Ranking-points difference

Using differences keeps the observation structure consistent across matches and makes each row directly comparable.

## Models

### Match Outcome Classification

Models evaluated:

- Dummy baseline
- Logistic Regression
- Polynomial Logistic Regression
- Random Forest Classifier
- Support Vector Machine
- Naive Bayes
- Multi-Layer Perceptron Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The current best-performing match-outcome model is Polynomial Logistic Regression with approximately 72.7% accuracy.

### Goal Difference Regression

Models evaluated:

- Dummy baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Regression
- Multi-Layer Perceptron Regressor

Evaluation metrics:

- Mean Absolute Error
- Root Mean Squared Error
- R-squared

### Tournament Success Classification

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest Classifier
- Multi-Layer Perceptron Classifier

Evaluation metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score

## 2026 World Cup Simulation

The Streamlit application includes a Monte Carlo-style 2026 World Cup simulator using the trained match-outcome model. The simulator includes:

- 48 qualified teams
- 12 groups
- Top two teams from each group advancing automatically
- Eight best third-place teams advancing
- Round of 32
- Round of 16
- Quarterfinals
- Semifinals
- Final
- Champion probability summary

Each simulated match uses model probability rather than a fixed prediction, allowing stronger teams to be favored while still preserving uncertainty.

## Current Results

The current saved metrics report:

- Best match model: Polynomial Logistic Regression
- Match-outcome accuracy: about 72.7%
- Training matches: 4,951
- Tournament-success rows: 126
- 2026 prediction-ready teams: 48 out of 48

The model performs meaningfully above the dummy baseline, which indicates that player-derived team strength, rankings, and comparative match features contain useful predictive signal.

## Limitations

The project has three important limitations:

- The local data does not include possession, shots, shots on target, or fouls, so those planned match-level variables are documented but not modeled.
- The requested 2018 World Cup squad file is not present, so roster strength is approximated from FIFA18 nationality-level player data rather than exact tournament squads.
- Jordan has FIFA ranking data but no FIFA18 player rows, so median player-strength fallback values are used for the 2026 simulator.

## Conclusion

This project develops a complete predictive pipeline from raw data to an interactive tournament simulator. It cleans and aligns multiple football datasets, engineers team-level and match-level variables, compares several classification and regression models, and applies the strongest match-outcome model to a forward-looking 2026 World Cup simulation.

The result is a rigorous and reproducible framework for understanding how player attributes, team composition, and ranking strength relate to international match outcomes and tournament progression.
