from nltk.stem import PorterStemmer
import string


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str, stopwords: list[str], stemmer: PorterStemmer) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens