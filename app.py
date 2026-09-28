import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import (
    DATASET_PATH,
    FEATURES,
    MODEL_PATH,
    evaluate_model,
    load_dataset,
    train_and_save_model,
)

# Page Configuration
st.set_page_config(
    page_title="Spotify Music Intelligence Dashboard",
    page_icon="🎵",
    layout="wide"
)

# Load Data and train the model on first launch if it has not been created yet.
df = load_dataset(DATASET_PATH)


@st.cache_resource
def get_model_and_metrics():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        r2, mae = evaluate_model(model, df)
        return model, r2, mae
    return train_and_save_model(DATASET_PATH, MODEL_PATH)


model, model_r2, model_mae = get_model_and_metrics()

# Header
st.title("🎵 Spotify Music Intelligence Dashboard")
st.markdown(
    "Analyze Spotify songs, artists, genres, and popularity trends using Data Science and Machine Learning."
)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Songs", f"{len(df):,}")

with col2:
    st.metric("Artists", f"{df['artists'].nunique():,}")

with col3:
    st.metric("Genres", f"{df['track_genre'].nunique():,}")

with col4:
    st.metric(
        "Avg Popularity",
        round(df["popularity"].mean(), 1)
    )

st.divider()

# Popularity Distribution
st.subheader("📈 Popularity Distribution")

fig_pop = px.histogram(
    df,
    x="popularity",
    nbins=30,
    title="Distribution of Song Popularity"
)

st.plotly_chart(fig_pop, use_container_width=True)

# Top Genres
st.subheader("🔥 Top Genres by Average Popularity")

genre_popularity = (
    df.groupby("track_genre")["popularity"]
      .mean()
      .sort_values(ascending=False)
      .head(15)
)

fig_genre = px.bar(
    x=genre_popularity.values,
    y=genre_popularity.index,
    orientation="h",
    title="Top Genres by Popularity"
)

st.plotly_chart(fig_genre, use_container_width=True)

# Top Artists
st.subheader("🎤 Top Artists by Average Popularity")

artist_popularity = (
    df.groupby("artists")["popularity"]
      .mean()
      .sort_values(ascending=False)
      .head(15)
)

fig_artist = px.bar(
    x=artist_popularity.values,
    y=artist_popularity.index,
    orientation="h",
    title="Top Artists"
)

st.plotly_chart(fig_artist, use_container_width=True)

# Scatter Plots
sample_df = df.sample(min(5000, len(df)), random_state=42)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💃 Danceability vs Popularity")

    fig_dance = px.scatter(
        sample_df,
        x="danceability",
        y="popularity"
    )

    st.plotly_chart(fig_dance, use_container_width=True)

with col2:
    st.subheader("⚡ Energy vs Popularity")

    fig_energy = px.scatter(
        sample_df,
        x="energy",
        y="popularity"
    )

    st.plotly_chart(fig_energy, use_container_width=True)

# Genre Explorer
st.subheader("🔍 Explore Songs by Genre")

selected_genre = st.selectbox(
    "Choose Genre",
    sorted(df["track_genre"].unique())
)

filtered_df = df[df["track_genre"] == selected_genre]

st.dataframe(
    filtered_df[
        [
            "track_name",
            "artists",
            "popularity",
            "danceability",
            "energy"
        ]
    ].head(20)
)

st.divider()

# Song Popularity Predictor
st.header("🤖 Song Popularity Predictor")

st.metric("Model R² Score", f"{model_r2:.3f}")
st.metric("MAE", f"{model_mae:.3f}")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50.0, 220.0, 120.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
liveness = st.slider("Liveness", 0.0, 1.0, 0.2)

if st.button("Predict Popularity"):

    prediction = model.predict(pd.DataFrame([[
        danceability,
        energy,
        valence,
        tempo,
        acousticness,
        speechiness,
        liveness
    ]], columns=FEATURES))

    st.success(
        f"Predicted Popularity Score: {prediction[0]:.1f}"
    )

st.divider()

# Feature Importance
st.header("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Audio Feature Importance"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)
