"""This script runs standard NLP Preprocessing on the given unclean dataset from Params"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
import spacy
import re

nlp = spacy.load("en_core_web_sm")

import pdb


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
        sentence = lemmatize(sentence, nlp)
        filtered_sentences.append(sentence)
    assert len(list(doc.sents)) == len(filtered_sentences)
    return filtered_sentences


def clean_sources(df):
    sources = list(df["site"].unique())
    sources_no_dot = [i.replace(".", "") for i in sources]
    sources_no_dotcom = [i.replace(".com", "") for i in sources]
    sources_no_news = [i.replace("news", "") for i in sources_no_dotcom]
    for i in sources_no_dot:
        sources.append(i)
    for i in sources_no_dotcom:
        sources.append(i)
    for i in sources_no_news:
        sources.append(i)
    sources.append("(cnn)")
    sources.append("washington")
    sources.append("(cnn business)")
    sources.append("new york")
    sources.append("(ap)")
    sources.append("(fox)")
    sources = list(dict.fromkeys(sources))
    df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sources))
    return df


def clean_doc(raw_path, write_path):
    """
    The following cleaning steps are performed on the dataset:
    1. Punctuation Removal
    2. Lowercase
    3. Stopword Removal
    4. Lemmatization
    """
    print("Reading raw data from: %s" % raw_path)
    print("Writing clean data to: %s" % write_path)

    df = pd.read_csv(raw_path)
    tqdm.pandas()

    # Removing Giveaway words about news sources:
    df = clean_sources(df)

    df["text"] = df["text"].progress_apply(normalize)

    # text_tokens = []
    # for index,row in tqdm(df.iterrows()):
    #     text_tokens.append(normalize(row['text'],nlp))
    # df['text'] = text_tokens
    df.to_csv(write_path, index=False)
    print("Clean dataset saved at location: %s" % write_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, help="location of raw data to be cleaned")
    parser.add_argument("--write_path", type=str, help="location to write clean data to")
    args = parser.parse_args()

    clean_doc(args.raw_path, args.write_path)

    """
    Footnotes.
    Optimize lemmatizer. Resource: https://towardsdatascience.com/turbo-charge-your-spacy-nlp-pipeline-551435b664ad 
    """
