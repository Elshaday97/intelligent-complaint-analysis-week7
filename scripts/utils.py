import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")
nltk.download("wordnet")


def clean_text(text: str) -> str:
    """
    Cleans the input text by performing several preprocessing steps:
    1. Converts text to lowercase.
    2. Removes URLs and hyperlinks.
    3. Strips HTML tags.
    4. Eliminates common boilerplate phrases.
    5. Removes special characters and punctuation.
    6. Normalizes whitespace.
    """
    # 1. Basic conversion and lowercasing
    text = str(text).lower()

    # 2. Remove URLs/Hyperlinks
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove HTML tags (common in scraped narratives)
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove boilerplate phrases
    # Add common narrative "fluff" to this list
    boilerplate = [
        r"i am writing to file a complaint regarding",
        r"to whom it may concern",
        r"please find attached",
        r"thank you for your time",
        r"i want to start out this complaint by stating",
        r"i had a friend help me write this complaint",
        r"i m writing to complain about",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text)

    # 5. Remove special characters and punctuation
    # We keep spaces and alphanumeric characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 6. Normalize Whitespace
    # Removes extra spaces, tabs, and newlines resulting from previous steps
    text = " ".join(text.split())

    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenizes and lemmatizes the input text.
    Args:
        text (str): The input text to process.
    Returns:
        str: The processed text after tokenization and lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize and remove stop words
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    return " ".join(processed_tokens)
