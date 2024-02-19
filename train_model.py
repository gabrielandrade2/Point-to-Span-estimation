import argparse
import json
import os

from sklearn.model_selection import train_test_split
from transformers import BertJapaneseTokenizer

from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list
from util.xml_parser import convert_xml_file_to_iob_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train from XML file')
    parser.add_argument('-i', '--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output folder', required=False)
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
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_type)
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
                                               device=args.device, tokenizer=tokenizer)
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

    print('Done')
