import json

import nltk  # type:ignore
import pandas as pd
from tqdm import tqdm  # type:ignore

import task1

nltk.download("stopwords")


def get_unique_pids() -> tuple[pd.Series, pd.Series]:
    """
    get_unique_pids read candidate-passages-top1000.tsv file
    and return unique passage ids (pid) and corresponding passages

    There are in total 182469 unique passages so the retunred series
    have the same lenght as well

    :return: 2 series. pid, passages
    :rtype: tuple[pd.Series, pd.Series]
    """

    candidate_passages_df = pd.read_table(
        "candidate-passages-top1000.tsv",
        delimiter="\t",
        header=None,
    )
    candidate_passages_df.columns = [
        "qid",
        "pid",
        "query",
        "passage",
    ]  # type:ignore

    candidate_passages_df = candidate_passages_df.drop_duplicates(
        subset=["pid"]
    )

    unique_pid = candidate_passages_df["pid"]
    unique_passage = candidate_passages_df["passage"]

    return unique_pid, unique_passage


def work_on_one_passage(
    unique_pid: int, unique_passage: str, inverted_index: dict
) -> dict:
    """
    work_on_one_passage Work on one passage at a time
    first function from task 1 is called to get a list of all
    tokens after preprocessing and stemming

    Then the inverted index is updated accordingly
    pid as well number of occureences in updated in the inverted index

    :param unique_pid: unique passage identifier
    :type unique_pid: int
    :param unique_passage: string passage
    :type unique_passage: str
    :param inverted_index: dictionary for the inverted index with all tokens
    in vocabulary as keys
    :type inverted_index: dict
    :return: updated inverted index
    :rtype: dict
    """

    stemmed_tokens = task1.work_one_line(unique_passage)

    for token in stemmed_tokens:
        try:
            # print(token, inverted_index[token])
            # print(unique_pid not in inverted_index[token])

            inverted_index[token].update(
                {
                    unique_pid: inverted_index[token].get(unique_pid, 0) + 1,
                }
            )
        except Exception as e:
            print(token)
            print(e)

    return inverted_index


def get_inverted_index():

    vocab_df_sorted = pd.read_csv(
        "passage_collection_stats.csv",
        na_filter=False,
    )
    inverted_index = {}
    for token in vocab_df_sorted["token"]:
        if token not in inverted_index:
            inverted_index[token] = {}
        else:
            # ideally it should never come here since
            # all tokens should be unique
            print(token, type(token))

    return inverted_index


if __name__ == "__main__":
    inverted_index = get_inverted_index()
    # print(inverted_index)
    unique_pids, unique_passages = get_unique_pids()
    # get all unique 182469 passages and thier ids

    for index, [pid, passage] in tqdm(
        enumerate(zip(unique_pids, unique_passages))
    ):
        inverted_index = work_on_one_passage(pid, passage, inverted_index)

        # if index == 1000:
        #     break

    # print(inverted_index)
    with open("inverted_index.json", "w") as f:
        json.dump(inverted_index, f)
        print("Done Saving inverted index as .json file")
