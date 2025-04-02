# Multi-Label Email Classification

This repository includes a multi-label email classification pipeline that processes CSV data, generates TF-IDF embeddings, and trains a machine learning model to classify emails into multiple categories. 

## Project Overview
- **main.py:** Orchestrates data loading, preprocessing, embedding generation, and model training.
- **preprocess.py:** Functions for deduplication, noise removal, and translating text to English.
- **embeddings.py:** Contains TF-IDF embedding creation methods.
- **Config.py:** Centralizes file paths and configuration variables.
- **utils.py:** (Optional) Placeholder for shared functionality (if needed).

## Getting Started

1. **Install Dependencies**  

2. **Run the Main Script**  

- Loads data from `AppGallery.csv` (or your designated file).
- Preprocesses the data (removes duplicates, cleans text, and translates non-English text).
- Generates TF-IDF embeddings.
- Trains a model (e.g., RandomForest) and saves the model to `final_model.pkl`.

## Contributors
- Your Name
- Other Team Members (if applicable)

## License
Choose a license (e.g., MIT, Apache 2.0) if needed.
