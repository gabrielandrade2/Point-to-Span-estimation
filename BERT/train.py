import argparse
import json
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertJapaneseTokenizer

from BERT import bert_utils
from BERT.Model import NERModel
from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import split_sentences, exclude_long_sentences
from util.xml_parser import convert_xml_file_to_iob_list


def train_from_xml_file(xmlFile, model_name, tag_list, output_dir, parameters=None, attr_list=None,
                        should_split_sentences=True, device=None):
    ##### Load the data #####
    sentences, tags = convert_xml_file_to_iob_list(xmlFile, tag_list, attr_list=attr_list,
                                                   should_split_sentences=should_split_sentences)
    return train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters, device)


def train_from_xml_texts(texts, model_name, tag_list, output_dir, parameters=None, attr_list=None,
                         should_split_sentences=True, device=None):
    if should_split_sentences:
        texts = split_sentences(texts)
    sentences, tags = convert_xml_text_list_to_iob_list(texts, tag_list, attr=attr_list)
    return train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters, device)


def train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters=None, local_files_only=False,
                                   device=None, validation_ratio=0.1, tokenizer=None):
    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=validation_ratio)
    return train_from_sentences_tags_list_val(train_x, train_y, validation_x, validation_y, model_name, output_dir,
                                              parameters, local_files_only, device, tokenizer)


def train_from_sentences_tags_list_val(train_x, train_y, validation_x, validation_y, model_name, output_dir,
                                       parameters=None, local_files_only=False,
                                       device=None, tokenizer=None):
    os.makedirs(output_dir, exist_ok=True)

    train_x, train_y = exclude_long_sentences(512, train_x, train_y)
    validation_x, validation_y = exclude_long_sentences(512, validation_x, validation_y)

    if not device:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print('device: ' + device)

    ##### Process dataset for BERT #####
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(train_y + validation_y)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, tokenizer, label_vocab)
    if validation_x and validation_y:
        validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, tokenizer,
                                                                      label_vocab)

    # Get pre-trained model and fine-tune it
    pre_trained_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_vocab))
    pre_trained_model.config.attention_probs_dropout_prob = 0.4  # Set the dropout probability for attention layers
    pre_trained_model.config.hidden_dropout_prob = 0.4  # Set the dropout probability for hidden laye
    pre_trained_model.resize_token_embeddings(len(tokenizer))
    model = NERModel(pre_trained_model, tokenizer, label_vocab, device=device)
    if validation_x and validation_y:
        model.train(train_x, train_y, parameters, val=[validation_x, validation_y], outputdir=output_dir)
    else:
        model.train(train_x, train_y, parameters, outputdir=output_dir)

    return model


def finetune_from_xml_file(xmlFile, model: NERModel, tag_list, output_dir, parameters=None, attr_list=None,
                           should_split_sentences=True):
    ##### Load the data #####
    sentences, tags = convert_xml_file_to_iob_list(xmlFile, tag_list, attr_list=attr_list,
                                                   should_split_sentences=should_split_sentences)
    return finetune_from_sentences_tags_list(sentences, tags, model, output_dir, parameters)


def finetune_from_xml_texts(texts, model: NERModel, tag_list, output_dir, parameters=None, attr_list=None,
                            should_split_sentences=True):
    if should_split_sentences:
        texts = split_sentences(texts)
    sentences, tags = convert_xml_text_list_to_iob_list(texts, tag_list, attr=attr_list)
    return train_from_sentences_tags_list(sentences, tags, model, output_dir, parameters)


def finetune_from_sentences_tags_list(sentences, tags, model: NERModel, output_dir=None, parameters=None,
                                      validation_ratio=0.1):
    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=validation_ratio)

    return finetune_from_sentences_tags_list(train_x, train_y, validation_x, validation_y, model, output_dir,
                                             parameters)


def finetune_from_sentences_tags_list_val(train_x, train_y, validation_x, validation_y, model: NERModel,
                                          output_dir=None,
                                          parameters=None):
    train_x, train_y = exclude_long_sentences(512, train_x, train_y)
    validation_x, validation_y = exclude_long_sentences(512, validation_x, validation_y)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, model.tokenizer, model.vocabulary)
    validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, model.tokenizer,
                                                                  model.vocabulary)

    # FineTune model
    model.train(train_x, train_y, parameters, val=[validation_x, validation_y], outputdir=output_dir)

    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(model.vocabulary, f, ensure_ascii=False)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train from XML file')
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--validation_ratio', type=float, help='Ratio of training data used in validation',
                        required=False)
    parser.add_argument('--test_file', type=str, help='Test file path', default=None)
    parser.add_argument('--test_ratio', type=float,
                        help='Ratio of training data used for testing if not test file is provided', default=0.2)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files')
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences')
    parser.add_argument('--device', type=str, help='Device', required=False, default="cpu")
    TrainingParameters.add_parser_arguments(parser)
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'

    # Load the training file
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char-v2')
    tokenizer.add_tokens(['⧫'])
    tokenizer.add_special_tokens({'additional_special_tokens': ['⧫']})

    sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr,
                                                   should_split_sentences=args.split_sentences, tokenizer=tokenizer.tokenize)

    # Check if a test file is provided
    if args.test_file is None:
        train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=args.test_ratio)
    else:
        test_x, test_y = convert_xml_file_to_iob_list(args.test_file, args.tags, attr_list=args.attr,
                                                      should_split_sentences=args.split_sentences, tokenizer=tokenizer.tokenize)
        train_x = sentences
        train_y = tags

    # Set training parameters
    parameters = TrainingParameters.from_args(args)

    if args.validation_ratio is not None:
        model = train_from_sentences_tags_list(train_x, train_y, model_type, args.output,
                                               parameters=parameters,
                                               local_files_only=args.local_files_only,
                                               validation_ratio=args.validation_ratio,
                                               device=args.device)
    else:
        model = train_from_sentences_tags_list(train_x, train_y, model_type, args.output,
                                               parameters=parameters,
                                               local_files_only=args.local_files_only,
                                               device=args.device, tokenizer=tokenizer)

    # Evaluate the model
    evaluate(model, test_x, test_y, save_dir=args.output, print_report=True, save_output_file=True)

    os.makedirs(args.output, exist_ok=True)
    with open(args.output + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
