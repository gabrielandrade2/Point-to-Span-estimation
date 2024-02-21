"""
This script is responsible for expanding the Point annotations from a dataset.

It applies a Point-to-Span model and predict the spans for the entity annotated with the ⧫ token.

Example usage:
    python expand_annotations.py -i datasets/dataset1.xml -o datasets/expanded/dataset1_expanded.xml -m models/point_to_span_model

@author: Gabriel Andrade
"""

import argparse
import os.path
import re

from tqdm import tqdm

from BERT.predict import main as predict_from_xml_file
from util.xml_parser import articles_to_xml


def remove_identifier_token(articles: list):
    for article in articles:
        article.text = article.text.replace('⧫', '')
    return articles


def exclude_extra_predictions(articles: list):
    """
    This function removes eventual erroneous spans predictions done by the model. These are identified by the lack of the ⧫ symbol in the span.
    """

    pattern = r'<C>(.*?)<\/C>'
    for article in articles:
        article.text = re.sub(pattern, lambda match: match.group() if '⧫' in match.group(1) else match.group(1), article.text)
    return articles


def detect_unexpanded_annotations(articles: list):
    """
    This function detects ⧫ tokens that were not expanded by the model, thus not enclosed by start and end span tags.
    There is no fix, so just the occurrences and statistics are printed.
    """

    xml_string = ''.join([article.text for article in articles])

    pattern = r'<C>(.*?)<\/C>|([^<]*?⧫[^>]*?)'
    matches = re.finditer(pattern, xml_string, re.DOTALL)

    total_symbols = xml_string.count('⧫')
    unclosed_symbols = 0

    for match in matches:
        if match.group(2):
            lines = xml_string.count('\n', 0, match.start(2)) + 1
            print(f"Found non-expanded ⧫ annotation on line {lines}: {match.group(2)}")
            unclosed_symbols += match.group(2).count('⧫')

    percentage = (unclosed_symbols / total_symbols) * 100 if total_symbols > 0 else 0

    print(f"\nTotal ⧫ annotations: {total_symbols}")
    print(f"Non-expanded ⧫ annotations: {unclosed_symbols}")
    print(f"Percentage of non-expanded annotations: {percentage:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies the expansion model to predict the spans of ⧫ annotations in the text'
                                                 'This script outputs a new XML file with the predicted spans')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='Input dataset file(s) in XML format', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output folder', required=True)
    parser.add_argument('-m', '--model', type=str, help='Model folder', required=True)
    parser.add_argument('--split_sentences', type=bool, help='Treat each sentence as a separate instance', required=False, default=True)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction, help='Use transformers local files')
    parser.add_argument('--device', type=str, help='Device to run model on', default=None, required=False)
    args = parser.parse_args()

    for file in tqdm(args.input, desc="Processing files"):
        processed_articles = predict_from_xml_file(args.model, file, args.split_sentences, args.local_files_only, args.device)
        processed_articles = exclude_extra_predictions(processed_articles)

        detect_unexpanded_annotations(processed_articles)

        processed_articles = remove_identifier_token(processed_articles)

        # Save to XML file
        print( os.path.join(args.output, os.path.basename(file)))
        articles_to_xml(processed_articles, os.path.join(args.output, os.path.basename(file)))

    print('Done')
