# Point-to-Span estimation BERT model

A model used to estimate the start and end of a Named Entity (NE) span based on a Point annotation,
as used in the paper "Is boundary annotation necessary? Evaluating boundary-free approaches to improve clinical named entity annotation efficiency".

This repository contains the codes for generating the training data based on an existing 
span-annotated dataset, training the estimation model and estimating the NE on a point-annotated 
dataset.

The trained model is available, as used in the paper, is available in the [Hugging Face model hub](https://huggingface.co/gabrielandrade2/point-to-span-estimation/tree/main).


Basically, the goal of this model is to convert a point annotation to a corresponding span annotation with the correct span.

In order to do so, the model is trained on a gold-standard span-annotated corpus, which is prepared by the `generate_training_data.py` script.
It adds an identifier token (♦) to each annotation. The model is then trained to predict the span of the annotation based on the context surrounding this token.

The trained model can then be used to estimate the NE span based on a point-annotated dataset, which does not have the span information about the concepts, using the `estimate-span.py` script.

## Concepts

### Point annotation

Unlike span-based paradigms, a point annotation is composed by a single position within the NE span.
It is a simple and fast way to annotate NEs, but it introduces ambiguity in the information captured by the annotation.

On this repository implementation, a point annotation is represented by a lozenge character (⧫).

Example:
```
The patient has a history of dia⧫betes.
```

### Span annotation

A span annotation is composed by the two markings, identifying both start and end positions of the NE span.

The implementation on this repository is based on the span annotation schema defined by [Yada et al. (2020)](https://aclanthology.org/2020.lrec-1.561/).

Example:
```
The patient has a history of <C>diabetes</C>.
```

## Model

The current implementation is based on the tohoku-nlp/bert-base-japanese-char-v2 and primarily designed for Japanese text.

However, the methodology used is language-agnostic and the code can be adapted to use other BERT models.

## Usage

`example.py` executes the complete the workflow with example parameters to run the scripts.

This repository contains three main scripts:

- `generate_training_data.py`

This script takes a span-annotated dataset and generates a training dataset for the estimation model, by adding an identifier token (♦) within the span of each annotation.

The position of the identifier token is determined by the `--strategy` parameter, which can be one of the following:

`random`: Places the identifier token in a random position within the span.
`gaussian`: Uses a truncated Gaussian distribution based on the span length to place the identifier token within the span.

An additional `--augmentation` parameter can be used to augment the training data by generating multiple variants of the same annotation with the identifier token placed in different positions.

Example usage:
```
python generate_training_data.py --input data/span-annotated-dataset.txt --output data/training-dataset.txt --strategy random --tags C
```

- `train_model.py`

Trains the estimation model based on the generated training data.

This script takes the training dataset generated by the previous step and trains a BERT model using it.

The default BERT used is the `tohoku-nlp/bert-base-japanese-char-v2`, but it can be changed by the `--base-model` parameter.

Additional arguments can be passed to control the model training/test parameters. Please check the script for more details.

Example usage:
```
python train_model.py --input data/training-dataset.txt --output model/point-to-span-estimation --tags C --max_epochs 10 --split_sentences
```

- `estimate-span.py`:

This script takes a point-annotated dataset and applies the trained estmation model to infer the spans for each one of the identified NE.

The output is a span-annotated dataset.

Example usage:
```
python estimate-span.py --input data/point-annotated-dataset.txt --output data/span-annotated-dataset.txt --model model/point-to-span-estimation
```

