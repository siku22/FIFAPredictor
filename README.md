# FIFAPredictor
Senior Capstone

## Local setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train/regenerate the local model artifacts:

```bash
python3 train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app expects these generated files in the project folder:

- `fifa_model.pkl`
- `team_features.pkl`
- `compiled_fifa_match_dataset.csv`
