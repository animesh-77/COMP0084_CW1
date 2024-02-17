import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # type:ignore
from nltk.stem import PorterStemmer  # type:ignore
from nltk.tokenize import word_tokenize  # type:ignore
from tqdm import tqdm  # type:ignore


def work_one_line(line: str) -> list:
    """
    work_one_line work on one line at a time and
    return all tokens present in it

    Steps taken in this function:
    1) convert everything to lower case
    2) All special characters replaced with " "
    3) All successuve " " removed
    4) String is then tokenized
    5) Porter stemmer used on individual tokens
    6) List of tokens after PorterStemmer has been used

    :param line: string value to be tokenized
    :type line: str
    :return: list of all tokens identified in the string
    :rtype: list
    """

    # convert whole line in lower case
    line = line.lower()

    # replace all kind of punctuation with " "
    line = re.sub(r"[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]", " ", line)

    # remove successive white spaces
    line = re.sub(r"\s+", " ", line)

    tokens = word_tokenize(line)

    tokens = [token for token in tokens if token.isalpha()]

    # print(tokens)

    # define Porter Stemmer from NLTK
    porter_stemmer = PorterStemmer()

    # stem tokenized text and print first 500 tokens
    stemmed_tokens = [porter_stemmer.stem(word) for word in tokens]
    # print(stemmed_tokens)

    return stemmed_tokens


def add_tokens(stemmed_tokens: list, all_tokens: dict) -> dict:
    """
    add_tokens from a list of tokens after stemming make a dictionary of
    counts of each token

    if token is already present in dictionary then increment count else
    make a new entry with count =1

    :param stemmed_tokens: list of tokens after stemming
    :type stemmed_tokens: list
    :param all_tokens: Dictionary of tokens and their counts observed so far
    :type all_tokens: dict
    :return: dictionary of updated tokens and theur counts
    :rtype: dict
    """
    for token in stemmed_tokens:
        if token not in all_tokens:
            all_tokens[token] = 1
        else:
            all_tokens[token] += 1

    return all_tokens


def get_all_tokens() -> dict:
    """
    get_all_tokens Read passage-collection.txt file and
    returns a dictionary of all tokens in vocabulary and their count

    _extended_summary_

    :return: dictionary of all words in vocabulary and their count
    :rtype: dict
    """
    all_tokens: dict = {}

    with open("passage-collection.txt", "r") as f:
        for line_num, line in enumerate(tqdm(f)):
            # print(line_num, line)
            stemmed_tokens = work_one_line(line)
            all_tokens = add_tokens(stemmed_tokens, all_tokens)
            # if line_num == 500:

            #     break

    return all_tokens


def get_vocab_df(all_tokens: dict) -> pd.DataFrame:
    """
    get_vocab_df from a dictionary of tokens and their count make a dataframe
    with follwing columns

    1) token / word
    2) Count
    3) Frequency of occurence / Prob of occurence
    4) Rank of word/ token starting from 1 for most frequent
    5) Empirical value of Frequency* rank

    :param all_tokens: _description_
    :type all_tokens: dict
    :return: _description_
    :rtype: pd.DataFrame
    """
    vocab_df = pd.DataFrame.from_dict(
        all_tokens,
        orient="index",
        columns=["Count"],
    )
    vocab_df = vocab_df.reset_index()
    vocab_df.rename(columns={"index": "token"}, inplace=True)

    vocab_df_sorted = vocab_df.sort_values(by=["Count"], ascending=False)
    vocab_df_sorted["rank"] = (vocab_df.index + 1).astype(float)

    total_voc = np.sum(vocab_df_sorted["Count"])
    vocab_df_sorted["Freq"] = vocab_df_sorted["Count"] / total_voc
    vocab_df_sorted["Freq*rank"] = (
        vocab_df_sorted["Freq"] * vocab_df_sorted["rank"]
    )
    vocab_df_sorted.to_csv("passage_collection_stats.csv", index=False)

    return vocab_df_sorted


def make_plots_task1(vocab_df_sorted: pd.DataFrame) -> None:

    Hn = np.sum(np.reciprocal(vocab_df_sorted["rank"]))

    zipf_freq = np.reciprocal(vocab_df_sorted["rank"] * Hn)

    plt.title("Zipf's Law Comparison")

    plt.xlabel("Frequence ranking of term")
    plt.ylabel("Prob of occurence of term")
    plt.plot(
        vocab_df_sorted["rank"],
        vocab_df_sorted["Freq"],
        color="blue",
        label="Data",
    )
    plt.plot(
        vocab_df_sorted["rank"],
        zipf_freq,
        linestyle="--",
        color="black",
        label="Zipf's curve",
    )
    plt.grid(True, which="major", ls="-")
    plt.legend()
    plt.savefig("Figure_1.svg")
    plt.show()

    plt.title("Zipf's Law Comparison in log scale")
    plt.xlabel("Frequence ranking of term")
    plt.ylabel("Prob of occurence of term")
    plt.loglog(
        vocab_df_sorted["rank"],
        vocab_df_sorted["Freq"],
        color="blue",
        label="Data",
    )
    plt.loglog(
        vocab_df_sorted["rank"],
        zipf_freq,
        linestyle="--",
        color="black",
        label="Zipf's curve",
    )
    plt.grid(True, which="major", ls="-")
    plt.legend()
    plt.savefig("Figure_2.svg")
    plt.show()


if __name__ == "__main__":

    # print(
    #     work_one_line(
    #         """A@This is a test sentence! It contains special characters:
    #             !\"#\$%&'(or)*+does,it-./:;<=>?@[\]^_`{|}~,"""
    #     )
    # )
    all_tokens = get_all_tokens()

    vocab_df_sorted = get_vocab_df(all_tokens)
    # print(vocab_df_sorted.head(20))

    make_plots_task1(vocab_df_sorted)
