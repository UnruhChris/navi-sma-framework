import re
import pandas as pd
from typing import Tuple
from collections import Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string
import emoji

# Funzione per rimuovere tag HTML
def remove_html_tags(texts: pd.Series) -> Tuple[pd.Series, dict]:
    def clean_html(text):
        if not isinstance(text, str):
            return text
        return BeautifulSoup(text, "html.parser").get_text()

    processed_texts = texts.apply(clean_html)
    changes = (texts != processed_texts).sum()
    stats = {"html_tags_removed": changes}
    return processed_texts, stats

# Funzione per rimuovere la punteggiatura
def remove_punctuation(texts: pd.Series) -> Tuple[pd.Series, dict]:
    def clean_punctuation(text):
        if not isinstance(text, str):
            return text
        return text.translate(str.maketrans('', '', string.punctuation))

    processed_texts = texts.apply(clean_punctuation)
    changes = (texts != processed_texts).sum()
    stats = {"punctuation_removed": changes}
    return processed_texts, stats

# Funzione per rimuovere hashtag e URL
def remove_hashtags_urls(texts: pd.Series) -> Tuple[pd.Series, dict]:
    def clean_hashtags_urls(text):
        if not isinstance(text, str):
            return text
        text = re.sub(r'http\S+', '', text)  # Rimuovi URL
        text = re.sub(r'#\w+', '', text)    # Rimuovi hashtag
        return text

    processed_texts = texts.apply(clean_hashtags_urls)
    changes = (texts != processed_texts).sum()
    stats = {"hashtags_urls_removed": changes}
    return processed_texts, stats

# Funzione per rimuovere stopword
def remove_stopwords(texts: pd.Series, language: str = 'english') -> Tuple[pd.Series, dict]:
    stop_words = set(stopwords.words(language))

    def clean_stopwords(text):
        if not isinstance(text, str):
            return text
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    processed_texts = texts.apply(clean_stopwords)
    changes = (texts != processed_texts).sum()
    stats = {"stopwords_removed": changes}
    return processed_texts, stats

# Funzione per rimuovere numeri
def remove_numbers(texts: pd.Series) -> Tuple[pd.Series, dict]:
    def clean_numbers(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'\d+', '', text)

    processed_texts = texts.apply(clean_numbers)
    changes = (texts != processed_texts).sum()
    stats = {"numbers_removed": changes}
    return processed_texts, stats

# Funzione per rimuovere emoji
def remove_emojis(texts: pd.Series) -> Tuple[pd.Series, dict]:
    def clean_emojis(text):
        if not isinstance(text, str):
            return text
        return emoji.replace_emoji(text, replace="")

    processed_texts = texts.apply(clean_emojis)
    changes = (texts != processed_texts).sum()
    stats = {"emojis_removed": changes}
    return processed_texts, stats
