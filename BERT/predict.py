import argparse

import torch
from tqdm import tqdm

from BERT.Model import NERModel
from util.iob_util import convert_list_iob_to_xml
from util.xml_parser import xml_to_articles, Article, articles_to_xml


def predict_from_sentences_list(model, sentences, split_sentences=False, display_progress=False):
    sentences_embeddings = model.prepare_sentences(sentences, split_sentences)
    labels = model.predict(sentences_embeddings, display_progress=display_progress)
    sentences = model.convert_ids_to_tokens(sentences_embeddings)
    labels = [[l if l != "[PAD]" else "O" for l in label] for label in labels]
    return sentences, labels


def main(model_path, input_file, split_sentences, local_files_only, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = NERModel.load_transformers_model(model_path, device, local_files_only)

    # Load XML file
    articles = xml_to_articles(input_file)

    # Predict
    processed_articles = []
    for article in tqdm(articles, desc='Predicting', total=len(articles), ascii=True, ncols=100):
        sentences, iob = predict_from_sentences_list(model, article.text, split_sentences=split_sentences,
                                                     display_progress=False)
        processed_text = convert_list_iob_to_xml(sentences, iob)
        processed_articles.append(Article(processed_text, article.headers))

    return processed_articles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict from XML file')
    parser.add_argument('--model_path', type=str, help='Model folder', required=True)
    parser.add_argument('--input_file', type=str, help='Input file path', default=None)
    parser.add_argument('--output_file', type=str, help='Output file path', default=None)
    parser.add_argument('--split_sentences', type=bool, help='Treat each sentence as a separate instance', required=False, default=True)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction, help='Use transformers local files')
    parser.add_argument('--device', type=str, help='Device to run model on', default=None, required=False)
    args = parser.parse_args()

    processed_articles = main(args.model_path, args.input_file, args.split_sentences, args.local_files_only, args.device)

    # Save to XML file
    articles_to_xml(processed_articles, args.output_file)
