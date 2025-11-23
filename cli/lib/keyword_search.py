from .text_utils import tokenize_text

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    results = []
    query_tokens = tokenize_text(query, stopwords, stemmer)
    for movie in movies:
        title_tokens = tokenize_text(movie["title"], stopwords, stemmer)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

