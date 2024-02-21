"""
This script adds the identifier ⧫ token to the text inside the span of the NER tags,

The main use of this script is to generate training data for the Point-to-Span expansion model.

A ⧫ token is added to the text inside the span of the NER tags, and the modified text is saved to a new XML file.
The position of the ⧫ token can be chosen to be determined randomly or by a truncated normal distribution, via the "distribution" parameter.
It is also possible to augment the dataset by duplicating texts and adding the ⧫ token in different positions. The number of duplications is determined by the "augmentation" parameter.

Usage example:
    To generate 10 augmented datasets with the ⧫ token added in random positions:
    python add_position_token.py -i datasets/dataset1.xml -o datasets/expanded/random ---tags C --augmentation 10 --distribution random

    To generate 1 augmented dataset with the ⧫ token added in positions determined by a truncated normal distribution:
    python add_position_token.py -i datasets/dataset1.xml -o datasets/expanded/gaussian -t C -a 1 -d gaussian

@author: Gabriel Andrade
"""
import argparse
import os
import random
import re
from enum import Enum

from scipy.stats import truncnorm
from tqdm import tqdm

from util.xml_parser import xml_to_articles, articles_to_xml, Article


class StrategyType(Enum):
    RANDOM = 'random'
    GAUSSIAN = 'gaussian'


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def add_token(text: str, type: StrategyType, divider: int):
    # Select a position to insert the token
    if type == StrategyType.RANDOM:
        position = random.randint(0, len(text))
    elif type == StrategyType.GAUSSIAN:
        position = round(get_truncated_normal(mean=len(text) / 2, sd=len(text) / divider, low=0, upp=len(text)).rvs())
    else:
        raise ValueError('Invalid distribution type')


    # Insert ⧫ at the position
    text = text[:position] + '⧫' + text[position:]

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data for the Point-to-Span expansion model.'
                                                 'This script basically adds the identifier ⧫ token to the text inside the span of the NER tags')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='Input dataset file(s) in XML format', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output folder', required=True)
    parser.add_argument('-t', '--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('-a', '--augmentation', type=int, help='Number of augmentations', required=False, default=1)
    parser.add_argument('-s', '--strategy', choices=[member.value for member in StrategyType], type=str, help='Token positioning strategy', required=False, default='random')
    parser.add_argument('--standard_deviation_divider', type=int, help='A value to control the standard deviation based on the annotation length', required=False, default=6)

    args = parser.parse_args()

    for file in tqdm(args.input, desc='Processing files'):
        processed_articles = []
        articles = xml_to_articles(file)

        for _ in range(args.augmentation):
            for article in tqdm(articles, desc='Processing articles', leave=False):
                for tag in args.tags:
                    # Find and modify text between the desired tags
                    modified_content = re.sub(r'<' + tag + r'>(.*?)<\/' + tag + r'>',
                                              lambda match: '<{}>'.format(tag) + add_token(match.group(1), StrategyType(args.strategy), args.standard_deviation_divider) + '</{}>'.format(tag),
                                              article.text,
                                              flags=re.DOTALL)

                    processed_articles.append(Article(modified_content, article.headers))

        filename = os.path.basename(file)
        articles_to_xml(processed_articles, os.path.join(args.output, filename))

    print('Done')
