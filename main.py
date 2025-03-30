import numpy as np
import random
import pandas as pd

from preprocess import de_duplication, noise_remover, translate_to_en
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict
from modelling.data_model import Data
from Config import Config

# Set a fixed seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    print("Loading dataset...")
    df = pd.read_csv(Config.APP_GALLERY_FILE)
    df.columns = df.columns.str.strip().str.lower()  # Ensure column names are standardized
    print("Dataset loaded successfully!")
    print("Columns in dataset:", df.columns.tolist())
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    if Config.TICKET_SUMMARY.lower() not in df.columns:
        print(f"Error: Column '{Config.TICKET_SUMMARY}' not found in the dataset.")
        exit(1)
    df = de_duplication(df)
    df = noise_remover(df)
    df.loc[:, Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    print("Data preprocessing completed!")
    return df
#
def get_embeddings(df):
    print("Generating text embeddings...")
    X = get_tfidf_embd(df)
    print("Embeddings generated successfully!")
    return X, df
#
def get_data_object(X, df):
    return Data(X, df)

def perform_modelling(data, df, name):
    print("Starting model training...")
    model_predict(data, df, name)
    print("Model training completed!")
#
if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)

    # Ensure required columns exist
    required_columns = [Config.INTERACTION_CONTENT.lower(), Config.TICKET_SUMMARY.lower()]
    df.columns = [col.lower() for col in df.columns]  # Standardize column names
    if any(col not in df.columns for col in required_columns):
        print("Error: Missing required columns in dataset.")
        exit(1)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    X, group_df = get_embeddings(df)
    data = get_data_object(X, df)

    perform_modelling(data, df, 'final_model')
