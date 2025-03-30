import pandas as pd
import re
from langdetect import detect
from googletrans import Translator
from nltk.corpus import stopwords

def de_duplication(df):
    """
    Removes duplicate entries based on 'ticket id' and 'interaction id'.
    """
    return df.drop_duplicates(subset=['ticket id', 'interaction id'])

def noise_remover(df):
    """
    Removes unnecessary noise (e.g., special characters, multiple spaces) from the 'interaction content' column.
    Also ensures that non-string values are handled.
    """
    stop_words = set(stopwords.words('english'))  # Remove common stopwords
    # df['interaction content'] = df['interaction content'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
    # df['interaction content'] = df['interaction content'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    # df['interaction content'] = df['interaction content'].apply(lambda x: re.sub(r'\s+', ' ', x))
    return df


def translate_to_en(text_list):
    """
    Translates a list of text into English if they are not in English.
    """
    translator = Translator()
    translated_texts = []

    for text in text_list:
        try:
            # Detect the language of the text
            detected_lang = detect(text)
            print(f"Detected language: {detected_lang}")

            # If the text is not in English, translate it
            if detected_lang != 'en':
                translated_text = translator.translate(text, src='auto', dest='en').text
                print(f"Translated text: {translated_text}")
            else:
                translated_text = text  # No translation needed if it's already in English

            translated_texts.append(translated_text)

        except Exception as e:
            print(f"Error translating text: {e}")
            translated_texts.append(text)  # Return the original text in case of error

    return translated_texts
