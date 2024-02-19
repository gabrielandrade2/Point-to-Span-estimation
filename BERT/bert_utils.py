""" BERT Tags"""
__CLS_TAG = '[CLS]'
__PAD_TAG = '[PAD]'

def normalize_dataset(sentences, tags, tokenizer):
    """ Use the tokenizer to apply the same normalization applied to full strings to a pre-tokenized dataset. It also
    adjusts the referring tags in case of removal/expansion of tokens.

    :param sentences: A list of list of character tokenized sentences.
    :param tags: A list of list of tags corresponding to the sentences.
    :param tokenizer: The tokenizer to be used to normalize the text.
    :return: Two lists, the processed sentences and the adjusted labels.
    """

    processed_sentences = list()
    processed_tags_sentences = list()

    for sentence, tag_sentence in zip(sentences, tags):
        processed_sentence = list()
        processed_tag_sentence = list()
        for character, tag_character in zip(sentence, tag_sentence):
            tokenized = tokenizer.tokenize(character)
            last_tag = str()
            for token in tokenized:
                if token == '' or token == ' ':
                    continue
                processed_sentence.append(token)

                # In the case we are expanding a character that is tagged with a Beggining tag, we make the subsequent
                # ones as Intra.
                if last_tag.startswith('B') and last_tag == tag_character:
                    tag_character = tag_character.replace('B', 'I', 1)
                processed_tag_sentence.append(tag_character)

        processed_sentences.append(processed_sentence)
        processed_tags_sentences.append(processed_tag_sentence)
    return processed_sentences, processed_tags_sentences

def create_label_vocab(labels):
    """ Creates a dictionary containing the label vocabulary of the dataset and creates and id for it.
        It iterates over all the label sequences of all the sentences of the dataset and keep all the unique tags.
        It also adds the control PAD tag.

    :param labels: The dataset labels.
    :return: A dictionary containing all unique tags the forms the dataset vocabulary and its corresponding id.
    """
    vocab = {}
    vocab[__PAD_TAG] = len(vocab)

    for sentence_label in labels:
        for label in sentence_label:
            if label not in vocab:
                vocab[label] = len(vocab)

    return vocab


def sentence_to_input_values(sent, tokenizer, add_tags=True):
    """Convert a sentence into the corresponding word embeddings to be passed to a BERT
    model.

    :param sent: A list of chars, representing a sentence.
    :param tokenizer: The tokenizer corresponding to the BERT model to be used.
    :param add_tags: Should it add BERT control tags (CLS)?
    :return: A list containing the word embeddings.
    """
    if add_tags:
        sent = [__CLS_TAG] + sent
    return tokenizer.convert_tokens_to_ids(sent)


def label_to_input_values(labels, vocabulary, add_tags=True):
    """Convert a list of labels into its corresponding id to be passed to a BERT model.

    :param labels: A list of tags, representing tag sequence of a sentence.
    :param vocabulary: A dictionary containing the used vocabulary.
    :param add_tags: Should it add BERT control tags (PAD)?
    :return: A list the converted label sequence to an id sequence.
    """
    if add_tags:
        labels = [__PAD_TAG] + labels
    return [vocabulary[token] if token in vocabulary else vocabulary[__PAD_TAG] for token in labels]


def dataset_to_bert_input(data_x, data_y, tokenizer, vocabulary, add_tags=True):
    """Given a dataset formed by a list of sentences and a list of tags, it converts the sentences into word embeddings
    and the labels into label ids,as can be interpreted by the BERT model.

    :param data_x: The sentences of the dataset. (Expects a list of list of characters).
    :param data_y: The labels for the sentences of the dataset.
    (Expects a list of list of labels for each character).
    :param tokenizer: The tokenizer to be used to generate the word embeddings.
    :param vocabulary: A dictionary containing the used vocabulary.
    :param add_tags: Should add model tags.
    :return:
    """
    bert_data_x = [sentence_to_input_values(x, tokenizer, add_tags) for x in data_x]
    bert_data_y = [label_to_input_values(x, vocabulary, add_tags) for x in data_y]

    return bert_data_x, bert_data_y


def prepare_sentences(sentences, tokenizer):
    """ Prepare sentences from model execution (tokenization + CLS tag addition).

    :param sentences: The list of sentences to be prepared.
    :param tokenizer: The tokenizer object used by the model.
    :return: The list of prepared sentences.
    """
    tokenized_sentences = [tokenizer.tokenize(t) for t in sentences]
    return [tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in tokenized_sentences]
