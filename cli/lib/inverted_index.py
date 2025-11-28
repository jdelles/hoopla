from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from pathlib import Path
from .text_utils import tokenize_text
from .search_utils import load_movies
import pickle

class InvertedIndex: 

    def __init__(self): 
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id, text):
        text_tokens = tokenize_text(text, [], self.stemmer)
        self.term_frequencies[doc_id].update(text_tokens)
        for token in text_tokens: 
            self.index[token.lower()].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index[term.lower()])
    
    def get_tf(self, doc_id, term):
        term_token = tokenize_text(term, [], self.stemmer)

        if len(term_token) != 1:
            raise ValueError("Can only get one term frequency")

        if doc_id not in self.term_frequencies:
            return 0
        
        return self.term_frequencies[doc_id].get(term_token[0], 0)
    
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
        
        with open('cache/term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open('cache/index.pkl', 'rb') as f:
                self.index = pickle.load(f)

            with open('cache/docmap.pkl', 'rb') as f:
                self.docmap = pickle.load(f)

            with open('cache/term_frequencies.pkl', 'rb') as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError: 
            raise FileNotFoundError("File not found. Must run build command first to create the index.")
            

            
            
    