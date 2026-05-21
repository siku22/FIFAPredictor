# FIFAPredictor

Senior capstone project for predicting FIFA match outcomes and simulating the 2026 FIFA World Cup using player attributes, team composition, rankings, and historical match results.

## Project Structure

```text
FIFAPredictor/
├── app/
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── data/
│   ├── raw/                      # Original source datasets
│   ├── processed/                # Generated modeling datasets
│   └── reference/                # Hand-curated reference files, including 2026 groups
├── models/                       # Saved model artifacts and metrics
├── notebooks/
│   └── FIFA.ipynb                # Original exploratory Colab notebook
├── docs/
│   └── CAPSTONE_REPORT.md        # Written capstone report draft
├── src/
│   └── train_model.py            # Reproducible training/evaluation pipeline
├── app.py                        # Convenience Streamlit launcher
├── requirements.txt
└── README.md
```

## Modeling Pipeline

The project follows this flow:

```text
raw datasets
→ cleaned team names
→ player-derived team features
→ match-level feature differences
→ match outcome, goal difference, and tournament success models
→ 2026 World Cup simulation
→ Streamlit app
```

The current training pipeline builds:

- Match outcome classifiers: Logistic Regression, Polynomial Logistic Regression, Random Forest, SVM, Naive Bayes, and MLP.
- Goal difference regressors: Linear Regression, Ridge, Lasso, Random Forest, SVR, and MLP.
- Tournament success classifiers: Logistic Regression, Decision Tree, Random Forest, and MLP.
- 2026 World Cup simulation using the trained match-outcome model probabilities.

Current saved model summary:

- Best match-outcome model: Polynomial Logistic Regression.
- Match-outcome accuracy: approximately 72.7%.
- Training matches: 4,951.
- Tournament-success rows: 126.
- 2026 teams available in the simulator: 48/48.

## Local Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train or regenerate all model artifacts:

```bash
python3 src/train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will be available at:

```text
http://localhost:8501
```

If another Streamlit process is already using port 8501, Streamlit may choose a later port such as `http://localhost:8502`.

## Main Outputs

Generated outputs are saved to:

- `models/fifa_model.pkl`
- `models/team_features.pkl`
- `models/model_metrics.pkl`
- `data/processed/compiled_fifa_match_dataset.csv`
- `data/processed/tournament_success_dataset.csv`

## Final Deliverables

- Interactive app: `app/streamlit_app.py`
- Launcher: `app.py`
- Training pipeline: `src/train_model.py`
- Written report draft: `docs/CAPSTONE_REPORT.md`
- Presentation deck: `docs/FIFA_World_Cup_Predictor_Capstone.pptx`
- Presentation outline and speaker notes: `docs/PRESENTATION_OUTLINE.md`
- App screenshots: `docs/assets/`
- Original notebook: `notebooks/FIFA.ipynb`

## Notes

- The notebook is preserved in `notebooks/` as the original exploration.
- The reproducible version of the project is now in `src/train_model.py` and `app/streamlit_app.py`.
- Possession, shots, shots-on-target, and fouls are listed as data gaps because those fields are not present in the local datasets.
- The requested `world_cup_2018_squads.xlsx` file is not present in the local project folder. Team strength is derived from FIFA18 nationality-level player records instead of exact 2018 roster membership.
- Jordan is included in 2026 predictions using ranking data plus median player-strength fallback features because it has no FIFA18 player rows.
