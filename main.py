import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("./data.csv")

# Data Preprocessing
df['Genre'] = df['Genre'].apply(lambda x: eval(x) if pd.notnull(x) else [])
df['Director'] = df['Director'].apply(lambda x: eval(x)[0] if pd.notnull(x) else "")
df['Stars'] = df['Stars'].apply(lambda x: eval(x) if pd.notnull(x) else [])

# One-hot encoding for genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genre'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Combine features
df = pd.concat([df, genre_df], axis=1)
features = ['Run Time in minutes', 'Votes', 'MetaScore', 'Movie Rating'] + list(mlb.classes_)
df[features] = df[features].fillna(0)

# Normalize features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# ANN Model
X = df[features]
similarity_scores = cosine_similarity(X)

# Create a sampled similarity dataset for training
from itertools import combinations
import random

pairs = list(combinations(range(len(X)), 2))
random.shuffle(pairs)
sampled_pairs = pairs[:50000]  # Use 50,000 pairs or adjust based on system capacity

X_pairs = np.array([(X.iloc[i].values + X.iloc[j].values) / 2 for i, j in sampled_pairs])
y = np.array([similarity_scores[i][j] for i, j in sampled_pairs])

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pairs, y, test_size=0.2, random_state=42)

# Define the ANN model
ann_model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Predict similarity as a value between 0 and 1
])

ann_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train ANN
ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Save the ANN model and data for Streamlit
ann_model.save("ann_model.h5")
df.to_csv("processed_movies.csv", index=False)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(mlb, "mlb.pkl")
