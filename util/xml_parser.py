import os

import lxml.etree as etree
import pandas as pd
from lxml.etree import XMLSyntaxError, XMLParser
from tqdm import tqdm

from util.iob_util import convert_xml_text_to_iob
from util.text_utils import *


class Article:

    def __init__(self, text, headers=None):
        self.text = text
        self.headers = headers

    def get_headers_as_str(self):
        if self.headers is None:
            return ''
        return ' '.join([f'{key}="{value}"' for key, value in self.headers.items()])


class ArticleReader:

    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        if next(self.file).strip() != '<articles>' and next(self.file).strip() != '<articles>':
            raise Exception("Unsupported file: {}", file_path)
        self.header_matcher = re.compile(r'([^ ]*)="?([^"]*)"?[ >]')

    def __iter__(self):
        return self

    def __next__(self):
        # Get next <article> entry
        line = ""
        while ('<article' not in line):
            line = next(self.file)
        lines = []
        headers = dict(self.header_matcher.findall(line))
        while (True):
            line = next(self.file)
            if line == '</article>\n':
                return Article(''.join(lines), headers)
            lines.append(line)


class IncrementalXMLReader:

    def __init__(self, file_path):
        self.parser = etree.iterparse(file_path, events=('end',), tag='article', recover=False)

    def __iter__(self):
        return self

    def __next__(self):
        _, elem = next(self.parser)
        text = self.__stringify_children(elem)
        text = text.strip()
        headers = dict(elem.attrib)
        return Article(text, headers)

    @staticmethod
    def __stringify_children(node):
        s = node.text
        if s is None:
            s = ''
        for child in node:
            temp = etree.tostring(child, encoding='unicode')
            if '<article' in temp:
                break
            s += temp
        return s


def xml_to_articles(file_path, return_iterator=False, type="ArticleReader"):
    """Extract all instances of <article> into a list of Article, from a given xml file.

    :param file_path: The path to the xml file.
    :param return_iterator: For huge files, it returns an iterator that read the file entry by entry instead of loading
    everything into memory.
    :return: List of Article, containing all the articles as found in the file.
    """
    print("Parsing XML file...")
    try:
        if type != "ArticleReader":
            raise Exception
        reader = ArticleReader(file_path)
    except Exception as e:
        print(e)
        reader = IncrementalXMLReader(file_path)
    if return_iterator:
        return reader
    else:
        return [text for text in tqdm(reader)]


def xml_to_article_texts(file_path, return_iterator=False):
    """Extract all instances of <article> into a text list, from a given xml file.

    :param file_path: The path to the xml file.
    :param return_iterator: For huge files, it returns an iterator that read the file entry by entry instead of loading
    everything into memory.
    :return: List of strings, containing all the articles as found in the file.
    """

    articles = xml_to_articles(file_path, return_iterator)
    if return_iterator:
        return (article.text for article in articles)
    else:
        return [article.text for article in articles]


def convert_xml_to_dataframe(file, tag_list, print_info=True):
    """ Converts a corpus xml file to a dataframe.

    :param file:
    :param tag_list:
    :param print_info:
    :return: The converted dataframe.
    """

    # Preprocess
    articles = xml_to_article_texts(file)
    articles = preprocessing(articles)
    f = open("out/iob.iob", 'w')
    article_index = 0
    processed_iob = list()
    for article in articles:
        iob = convert_xml_text_to_iob(article, tag_list)
        f.write('\n'.join('{}	{}'.format(x[0], x[1]) for x in iob))
        for item in iob:
            processed_iob.append((article_index,) + item)
        f.write('\n')
        article_index = article_index + 1
    df = pd.DataFrame(processed_iob, columns=['Article #', 'Word', 'Tag'])

    # Print some info
    if print_info:
        print(df.head())
        print("Number of tags: {}".format(len(df.Tag.unique())))
        print(df.Tag.value_counts())

    return df


def convert_xml_file_to_iob_list(file, tag_list=None, attr_list=None, should_split_sentences=False,
                                 ignore_mismatch_tags=True, tokenizer=list):
    """Converts a corpus xml file to a tuple of strings and IOB tags.
    The strings can be split by article or sentences.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param should_split_sentences: Should articles be split into sentences?
    :param ignore_mismatch_tags: Should skip texts that contain missing/mismatch xml tags
    :return: The list of strings and the list of IOB tags
    """

    # Preprocess
    texts = __prepare_texts(file, should_split_sentences)

    # Convert
    items = list()
    tags = list()
    i = 0
    for t in texts:
        sent = list()
        tag = list()
        try:
            iob = convert_xml_text_to_iob(t, tag_list, attr_list, ignore_mismatch_tags=ignore_mismatch_tags,
                                          tokenizer=tokenizer)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
        except XMLSyntaxError:
            print("Skipping text with xml syntax error, id: " + str(i))
        i = i + 1
    return items, tags


def convert_xml_file_to_iob_file(file, tag_list, out_file, ignore_mismatch_tags=True):
    """Converts a corpus xml file into IOB2 format and save it to a file in CONLL 2003 format.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param out_file: The output path for the .iob file
    :param ignore_mismatch_tags: Should skip texts that contain missing/mismatch xml tags
    """

    # Preprocess
    texts = __prepare_texts(file, False)
    texts = split_sentences(texts, False)

    if not out_file.endswith('.iob'):
        out_file.append('iob')

    try:
        f = open(out_file, 'w')
    except OSError:
        print("Failed to open file for writing: " + out_file)
        return
    for text in texts:
        for sentence in text:
            try:
                iob = convert_xml_text_to_iob(sentence, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
                f.write('\n'.join('{}\t{}'.format(x[0], x[1]) for x in iob))
                f.write('\n\n')
            except XMLSyntaxError:
                print("Skipping sentence with xml syntax error")


def __prepare_texts(file, should_split_sentences):
    """ Loads a file and applies all the preprocessing steps before format conversion.

    :param file: The xml file to be loaded.
    :param should_split_sentences: Should the sentences from an article be split.
    :return: The list of string with the desired format.
    """
    articles = xml_to_article_texts(file)
    articles = preprocessing(articles)

    if should_split_sentences:
        texts = split_sentences(articles)
    else:
        texts = articles
    return texts


def drop_texts_with_mismatched_tags(texts):
    no_mismatch = list()
    for text in texts:
        try:
            tagged_text = '<sent>' + text + '</sent>'
            parser = XMLParser()
            parser.feed(tagged_text)
            no_mismatch.append(text)
        except XMLSyntaxError:
            continue
    return no_mismatch


def articles_to_xml(articles: list[Article], file_path: os.path):
    """
    Write a list of articles to an xml file.

    :param articles: The list of articles to be written.
    :param file_path: The path of the file to be written.
    """

    dir = os.path.dirname(file_path)
    if dir:
        os.makedirs(dir, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<articles>\n')
        for article in articles:
            headers = article.get_headers_as_str()
            if (len(headers) > 0):
                f.write('<article {}>\n'.format(headers))
            else:
                f.write('<article>\n')
            f.write(article.text)
            f.write('\n</article>\n')
        f.write('</articles>')


def convert_point_to_position(text, tag):
    tag = r"<{}/>".format(tag)
    text = re.sub(r"({})+".format(tag), tag, text)
    annotations = [m.start() for m in re.finditer(tag, text)]
    annotations = [a - k * len(tag) for k, a in enumerate(annotations)]
    return annotations
