# Spotify Music Intelligence Dashboard

A Streamlit dashboard for exploring Spotify track popularity, genres, artists, and audio features. It also trains a random forest model to estimate a track's popularity from seven audio features.

## Run locally

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On its first launch, the app trains and saves `spotify_model.pkl` beside the scripts. Later launches reuse the saved model. The dataset file `dataset.csv` must stay in this folder.

To train or evaluate the model separately:

```bash
python train_model.py
python feature_importance.py
```

The feature importance script saves `feature_importance.png` in this folder.
