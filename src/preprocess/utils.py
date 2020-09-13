# IMPORT STUFF

import datetime, pickle, os, codecs, re, string
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import pdb

import string
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
import spacy
import re

nlp = spacy.load("en_core_web_sm")

# GENERAL PURPOSE UTILS
class Params:
    """A utility class to load hyperparameters from a json file"""

    def __init__(self, json_path):
        self.update = json_path

        # load parameters onto members
        self.load(self.update)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as jf:
            json.dump(self.__dict__, jf, indent=4)

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as jf:
            params = json.load(jf)
            self.__dict__.update(params)

    @property
    def dict(self):
        """give the class dictionary-like access (e.g. params.dict['learning_rate']) """
        return self.__dict__


def get_dataset(file_path, batch_size, **kwargs):
    """Data loader from dataset csv file
    Args:
        file_path: (str) path to dataset
        batch_size: (int) Batch size
    Return:
        dataset: (tf.data.Dataset)
    """

    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        select_columns=["text", "label"],
        batch_size=batch_size,
        label_name="label",
        ignore_errors=True,
        shuffle=True,
        **kwargs,
    )
    return dataset


def fix_fn(features, label):
    """Transforms (tf.data.make_csv_dataset) to (tf.tfds.data.dataset)
    Args:
        features: (oDict) From csv dataset object
        labels: (tf.int32) From csv dataset object
    Return:
        features: (tf.string) Basically maps this from oDict to string
        label: (tf.int32)
    """

    def _fix(text, label):
        text = text.numpy()
        return text, label

    return tf.py_function(_fix, inp=[features["text"], label], Tout=(tf.string, tf.int32))


def _py_fn(fn):
    def apply_tf_pyfunc(*args, **kwargs):
        return tf.py_function(fn, list(args), **kwargs)

    return apply_tf_pyfunc


# NLP UTILS
def load_subword_embedding(word_index, emb_path, save_path=None):
    """Creates an embedding matrix from a pretrained word embedding
    Args:
        word_index: (tf.keras.utils.preprocessing.Tokenizer.word_index)
        emb_path: (str) path to the pretrained word embedding
    Return:
        embedding_matrix: (np.array, shape = (N, like word embedding), dtype = float32)
    """

    if not os.path.exists(emb_path):
        os.system('wget "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"')
        os.system("unzip glove.840B.300d.zip")
        os.system("rm glove.840B.300d.zip")
        emb_path = os.getcwd() + "/glove.840B.300d.txt"
        print("Downloaded and saved GloVe Word Embeddings to %s" % emb_path)

    embeddings_index = {}
    f = codecs.open(emb_path, encoding="utf-8")
    for line in tqdm(f, "loading subword embedding"):
        values = line.rstrip().rsplit(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors" % len(embeddings_index))

    # embedding matrix
    print("Preparing embedding matrix...")
    words_not_found = []

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    if save_path != None:
        with open(save_path, "wb") as fp:
            np.save(fp, embedding_matrix)
        print("Embedding Matrix saved as %s" % save_path)

    return embedding_matrix


def encode_texts(params, texts, tokenizer):
    """Encodes texts as they come from the data pipeline to go into model
    Args:
        params: (Params) Params used here are MAX_SENTENCE_COUNT, MAX_SENTENCE_LENGTH
        texts: (tf.string) path to the pretrained word embedding
        tokenizer: (tf.keras.utils.Preprocessing.Tokenizer) Tokenizer which has been built on the dataset
    Return:
        embedding_matrix: (np.array, shape = (N, like word embedding), dtype = float32)
    """
    texts = np.char.decode(texts.numpy().astype(np.bytes_), "UTF-8")
    texts = [texts.tolist()]
    encoded_texts = np.zeros((len(texts), params.MAX_SENTENCE_COUNT, params.MAX_SENTENCE_LENGTH))
    for i, text in enumerate(texts):
        encoded_text = np.array(
            tf.keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences(text), maxlen=params.MAX_SENTENCE_LENGTH
            )
        )[: params.MAX_SENTENCE_COUNT]
        encoded_texts[i][-len(encoded_text) :] = encoded_text
    return encoded_texts


def clean_puncts(x):
    x = str(x)
    puncts = [
        ",",
        ".",
        '"',
        ":",
        ")",
        "(",
        "-",
        "!",
        "?",
        "|",
        ";",
        "'",
        "$",
        "&",
        "/",
        "[",
        "]",
        ">",
        "%",
        "=",
        "#",
        "*",
        "+",
        "\\",
        "•",
        "~",
        "@",
        "£",
        "·",
        "_",
        "{",
        "}",
        "©",
        "^",
        "®",
        "`",
        "<",
        "→",
        "°",
        "€",
        "™",
        "›",
        "♥",
        "←",
        "×",
        "§",
        "″",
        "′",
        "Â",
        "█",
        "½",
        "à",
        "…",
        "“",
        "★",
        "”",
        "–",
        "●",
        "â",
        "►",
        "−",
        "¢",
        "²",
        "¬",
        "░",
        "¶",
        "↑",
        "±",
        "¿",
        "▾",
        "═",
        "¦",
        "║",
        "―",
        "¥",
        "▓",
        "—",
        "‹",
        "─",
        "▒",
        "：",
        "¼",
        "⊕",
        "▼",
        "▪",
        "†",
        "■",
        "’",
        "▀",
        "¨",
        "▄",
        "♫",
        "☆",
        "é",
        "¯",
        "♦",
        "¤",
        "▲",
        "è",
        "¸",
        "¾",
        "Ã",
        "⋅",
        "‘",
        "∞",
        "∙",
        "）",
        "↓",
        "、",
        "│",
        "（",
        "»",
        "，",
        "♪",
        "╩",
        "╚",
        "³",
        "・",
        "╦",
        "╣",
        "╔",
        "╗",
        "▬",
        "❤",
        "ï",
        "Ø",
        "¹",
        "≤",
        "‡",
        "√",
        "#",
        "—–",
        "cnn",
        "afp",
        "fox",
    ]
    for punct in puncts:
        x = x.replace(punct, f" {punct} ")
    return x


def clean_str(string):
    string = re.sub("cnn", "", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    cleanr = re.compile("<.*?>")
    string = re.sub(cleanr, "", string)
    string = string.replace("_", "")

    return string.strip().lower()


def clean_stopwords(sentence, nlp):
    sent = nlp(sentence)
    filtered_sentence = []
    for word in sent:
        lexeme = nlp.vocab[str(word)]
        if lexeme.is_stop == False:
            filtered_sentence.append(str(word))

    return " ".join(filtered_sentence)


def lemmatize(sentence, nlp):
    sent = nlp(sentence)

    return " ".join([token.lemma_ for token in sent])


def normalize(text):
    doc = nlp(text)
    token_list = [token.text for token in doc]
    filtered_sentences = []
    for sentence in doc.sents:
        sentence = clean_puncts(sentence)
        sentence = clean_str(sentence)
        sentence = clean_stopwords(sentence, nlp)
        # sentence = lemmatize(sentence, nlp)
        filtered_sentences.append(sentence)
    assert len(list(doc.sents)) == len(filtered_sentences)
    return filtered_sentences


# TRAIN-VAL SPLIT
def _train_validate_test_split(df, train_percent=0.8, validate_percent=0.2, seed=243):
    """Helper function for build_train_test()"""
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    return train, validate


def build_train_test(main_csv_path, dataset_subset_ratio, seed=243):
    """This function creates the train, validation, and test csv files"""
    """The created csvs are be what should be passed into input_fn() to create the tf.data.dataset object"""
    df = pd.read_csv(main_csv_path)
    # Shrink dataset by the ratio provided
    df = df.sample(frac=dataset_subset_ratio, random_state=seed).reset_index(drop=True)
    train, val = _train_validate_test_split(df, train_percent=0.8, validate_percent=0.2)
    train.to_csv("train.csv", index=False)
    print("Created train.csv in current directory")
    val.to_csv("val.csv", index=False)
    print("Created val.csv in current directory")
