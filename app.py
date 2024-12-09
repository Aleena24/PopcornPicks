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
def recommend_movies(movie_name, top_n=5):
    if movie_name not in df['Movie Name'].values:
        return "Movie not found in the dataset!"
    
    movie_idx = df[df['Movie Name'] == movie_name].index[0]
    movie_vector = df[features].iloc[movie_idx].values.reshape(1, -1)
    similarities = cosine_similarity(movie_vector, df[features]).flatten()
    
    top_indices = similarities.argsort()[-top_n-1:-1][::-1]
    recommendations = df.iloc[top_indices][['Movie Name', 'Movie Rating', 'Genre']]
    return recommendations

# Streamlit App
st.title("Movie Recommendation System")
st.write("Type a movie name and get recommendations based on genres and ratings.")

# User Input
movie_name = st.text_input("Enter Movie Name:")

if st.button("Recommend"):
    try:
        if movie_name:
            recommendations = recommend_movies(movie_name)
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.write("Top Recommendations:")
                st.table(recommendations)
        else:
            st.warning("Please enter a movie name.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
