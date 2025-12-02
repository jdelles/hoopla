#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex
from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build reusable index of movies")

    tf_parser = subparsers.add_parser("tf", help="get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="The document id you're searching")
    tf_parser.add_argument("term", type=str, help="A term you're searching")

    idf_parser = subparsers.add_parser("idf", help="get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="The term you're checking")

    tfidf_parser = subparsers.add_parser("tfidf", help="get tfidf")
    tfidf_parser.add_argument("doc_id", type=int, help="the document id you're checking")
    tfidf_parser.add_argument("term", type=str, help="the term you're checking")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="The number of results you would like")

    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "bm25search":
            index.load()
            bm25search = index.bm25_search(args.query, args.limit)
            for doc_id, score in bm25search:
                title = index.get_title_by_id(doc_id)
                print(f"({doc_id}) {title} - Score: {score:.2f}")
        case "bm25tf":
            index.load()
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25idf":
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "build":
            index.build()
            index.save()
        case "idf":
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for res in results:
                print(f"{res['id']} {res['title']}")
        case "tf":
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case "tfidf":
            index.load()
            tf_idf = index.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
