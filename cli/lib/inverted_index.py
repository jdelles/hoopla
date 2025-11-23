from collections import defaultdict
from nltk.stem import PorterStemmer
from pathlib import Path
from .text_utils import tokenize_text
from .search_utils import load_movies
import pickle

class InvertedIndex: 

    def __init__(self): 
        self.index = defaultdict(set)
        self.docmap = {}
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id, text):
        text_tokens = tokenize_text(text, [], self.stemmer)
        for token in text_tokens: 
            self.index[token.lower()].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index[term.lower()])
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        Path('cache').mkdir(exist_ok=True)
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)

            
            
    