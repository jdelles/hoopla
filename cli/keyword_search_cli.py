#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build reusable index of movies")

    tf_parser = subparsers.add_parser("tf", help="get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="The document id you're searching")
    tf_parser.add_argument("term", type=str, help="A term you're searching")

    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "build":
            index.build()
            index.save()
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for res in results:
                print(f"{res['id']} {res['title']}")
        case "tf":
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
