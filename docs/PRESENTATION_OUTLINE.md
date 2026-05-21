# Presentation Outline

## Slide 1: Title

Predicting FIFA World Cup Outcomes Using Player Attributes, Team Composition, and Match-Level Analysis.

Speaker note: Introduce the project as an end-to-end pipeline that starts with raw football data and ends with an interactive prediction and simulation app.

## Slide 2: Research Question

Main question: can player attributes, rankings, and team composition explain match outcomes and support a forward-looking tournament simulation?

Speaker note: Emphasize that the goal is not just one prediction, but a reproducible modeling framework.

## Slide 3: Data Foundation

Datasets used:

- FIFA18 player attributes
- International match results
- FIFA rankings
- World Cup match history
- Country-name mappings
- 2026 World Cup group reference data

Speaker note: Mention the documented limitations: exact 2018 squad file and possession/shots/fouls data were not present locally.

## Slide 4: Feature Engineering

Team-level features are built from player data and rankings, then converted into match-level differences.

Speaker note: Explain why differences matter: every row compares two teams directly, which makes model inputs consistent.

## Slide 5: Modeling Tests

Three tasks:

- Match outcome classification
- Goal difference regression
- Tournament success classification

Speaker note: The project compares several models instead of relying on a single algorithm.

## Slide 6: Match Model Results

Best match model: Polynomial Logistic Regression.

Current accuracy: approximately 72.7%.

Speaker note: Point out that Polynomial Logistic Regression is meaningfully above the dummy baseline, showing the engineered features have predictive signal.

## Slide 7: Interactive Match Predictor

Shows the Streamlit match predictor where users select two teams and get win probabilities.

Speaker note: This connects the model to a usable interface.

## Slide 8: 2026 World Cup Simulation

Shows repeated tournament simulations across the 48-team, 12-group 2026 format.

Speaker note: The simulation uses probabilities, so outcomes reflect uncertainty rather than one fixed bracket.

## Slide 9: Model Report Evidence

Shows model comparison tables, confusion matrix, feature importance, and known data gaps.

Speaker note: This is the transparency slide. It shows what was tested and what limitations remain.

## Slide 10: Conclusion

Final message:

- Built a complete raw-data-to-dashboard pipeline.
- Polynomial Logistic Regression achieved the strongest match-outcome performance.
- The app supports real-time prediction and tournament simulation.
- Future work should add exact roster files and richer in-game match statistics.

Speaker note: Close by tying the project back to the original goal: using connected data sources to understand and forecast World Cup performance.
