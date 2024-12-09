import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load saved models and data
ann_model = load_model("ann_model.h5")
df = pd.read_csv("processed_movies.csv")
scaler = joblib.load("scaler.pkl")
mlb = joblib.load("mlb.pkl")

# Normalize features
features = ['Run Time in minutes', 'Votes', 'MetaScore', 'Movie Rating'] + list(mlb.classes_)
df[features] = scaler.transform(df[features])

# Recommendation Function
def recommend_movies(movie_name, top_n=5, filters=None):
    movie_name_lower = movie_name.lower()
    matching_movies = df[df['Movie Name'].str.lower() == movie_name_lower]

    if matching_movies.empty:
        return "Movie not found in the dataset!"
    
    movie_idx = matching_movies.index[0]
    movie_vector = df[features].iloc[movie_idx].values.reshape(1, -1)
    similarities = cosine_similarity(movie_vector, df[features]).flatten()
    
    # Sort by similarity and IMDb rating
    df['similarity'] = similarities
    filtered_df = df.drop(index=movie_idx)  # Exclude the selected movie

    if filters:
        if 'Year' in filters and filters['Year']:
            filtered_df = filtered_df[filtered_df['Year of Release'] == filters['Year']]
        if 'Director' in filters and filters['Director']:
            filtered_df = filtered_df[filtered_df['Director'].str.contains(filters['Director'], case=False, na=False)]
        if 'Stars' in filters and filters['Stars']:
            filtered_df = filtered_df[filtered_df['Stars'].apply(lambda x: any(star in x for star in filters['Stars']))]
    
    top_recommendations = (
        filtered_df.sort_values(by=['similarity', 'Movie Rating'], ascending=[False, False])
        .head(top_n)
        [['Movie Name', 'Movie Rating', 'Genre', 'Year of Release']]
    )
    return top_recommendations

# Streamlit App
st.title("Movie Recommendation System")
st.write("Get personalized movie recommendations!")

# User Input
movie_name = st.text_input("Enter Movie Name:")
filter_year = st.number_input("Year of Release (Optional):", min_value=1900, max_value=2100, step=1, value=None)
filter_director = st.text_input("Director Name (Optional):")
filter_stars = st.text_input("Actor/Actress Name(s) (Optional, comma-separated):")
filter_stars_list = [s.strip() for s in filter_stars.split(',')] if filter_stars else []

if st.button("Recommend"):
    try:
        filters = {
            'Year': int(filter_year) if filter_year else None,
            'Director': filter_director,
            'Stars': filter_stars_list
        }
        recommendations = recommend_movies(movie_name, top_n=5, filters=filters)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write("Top Recommendations:")
            st.table(recommendations)
    except Exception as e:
        st.error(f"An error occurred: {e}")
