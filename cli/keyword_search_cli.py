#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build reusable index of movies")

    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for res in results:
                print(f"{res['id']} {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
