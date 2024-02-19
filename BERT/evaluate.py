import json
import os

from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
from seqeval.scheme import IOB2

from BERT.Model import NERModel
from util import iob_util
from util.list_utils import list_size
from util.relaxed_metrics import calculate_relaxed_metric
from util.text_utils import exclude_long_sentences


def evaluate(model: NERModel, test_sentences: list, test_labels: list, save_dir: str = None, print_report: bool = True,
             save_output_file: bool = False):
    # Convert to BERT data model
    # test_x, test_y = bert_utils.dataset_to_bert_input(test_sentences, test_labels, model.tokenizer, model.vocabulary)

    test_sentences, test_labels = exclude_long_sentences(512, test_sentences, test_labels)

    # Predict outputs
    sentences = model.prepare_sentences(test_sentences)
    predicted_labels = model.predict(sentences, display_progress=True)
    sentences = model.convert_ids_to_tokens(sentences)
    labels = [[l if l != "[PAD]" else "O" for l in label] for label in predicted_labels]

    # Normalize to same tokenization as BERT
    test_sentences, test_labels = model.normalize_tagged_dataset(test_sentences, test_labels)

    # Evaluate model
    if not (list_size(test_sentences) == list_size(sentences) == list_size(test_labels) == list_size(predicted_labels)):
        tmp_gl = []
        tmp_tl = []
        for gs, gl, ts, tl in zip(test_sentences, test_labels, sentences, predicted_labels):
            if len(gs) == len(gl) == len(ts) == len(tl):
                tmp_gl.append(gl)
                tmp_tl.append(tl)
                continue
            print("Sentence length mismatch")
            print(len(gs), len(gl), len(ts), len(tl))
            print("GS: ", gs)
            print("TS: ", ts)
        test_labels = tmp_gl
        predicted_labels = tmp_tl

    metrics = {
        'accuracy': accuracy_score(test_labels, predicted_labels),
        'precision': precision_score(test_labels, predicted_labels),
        'recall': recall_score(test_labels, predicted_labels),
        'f1': f1_score(test_labels, predicted_labels),
        'report': classification_report(test_labels, predicted_labels, scheme=IOB2)
    }

    relaxed_results = calculate_relaxed_metric(test_labels, predicted_labels)

    metrics["overall_f1_relaxed"] = relaxed_results["overall"]["f1"]
    metrics["overall_precision_relaxed"] = relaxed_results["overall"]["precision"]
    metrics["overall_recall_relaxed"] = relaxed_results["overall"]["recall"]

    if print_report:
        print('Accuracy: ' + str(metrics['accuracy']))
        print('Precision: ' + str(metrics['precision']))
        print('Recall: ' + str(metrics['recall']))
        print('F1 score: ' + str(metrics['f1']))
        print('Relaxed Precision: ' + str(metrics["overall_precision_relaxed"]))
        print('Relaxed Recall: ' + str(metrics["overall_recall_relaxed"]))
        print('Relaxed F1: ' + str(metrics["overall_f1_relaxed"]))
        print(metrics['report'])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/test_metrics.txt', 'w') as f:
            json.dump(metrics, f, indent=4)

        if save_output_file:
            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))

            with open(save_dir + '/output_file.txt', 'w') as f:
                f.write("\n".join(tagged_sentences))

    metrics.pop('report')
    return metrics
