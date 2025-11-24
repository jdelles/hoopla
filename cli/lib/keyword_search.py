from .text_utils import tokenize_text

from .search_utils import DEFAULT_SEARCH_LIMIT, load_stopwords
from nltk.stem import PorterStemmer
from lib.inverted_index import InvertedIndex

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    results = []
    query_tokens = tokenize_text(query, stopwords, stemmer)
    for token in query_tokens:
        indices = index.get_documents(token)
        for idx in indices:
            results.append(index.docmap[idx])
            if len(results) >= limit:
                break

    return results
