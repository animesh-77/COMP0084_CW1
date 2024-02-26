import pickle

import nltk  # type:ignore
import pandas as pd
from tqdm import tqdm  # type:ignore

import task1


def get_unique_pids() -> tuple[pd.Series, pd.Series]:
    """
    get_unique_pids read candidate-passages-top1000.tsv file
    and return unique passage ids (pid) and corresponding passages

    There are in total 182469 unique passages so the returned series
    have the same length as well

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
    unique_pid: int,
    unique_passage: str,
    inverted_index: dict,
    doc_lengths: dict,
    stop_tokens: dict,
) -> tuple[dict, dict]:
    """
    work_on_one_passage Work on one passage at a time
    1) function from task 1 is called to get a list of all
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

    stemmed_tokens = task1.work_one_line_no_stopwords(
        unique_passage, stop_tokens
    )
    doc_lengths[unique_pid] = len(stemmed_tokens)
    for token in stemmed_tokens:
        try:
            # print(token, inverted_index[token])
            # print(unique_pid not in inverted_index[token])

            inverted_index[token].update(
                {
                    unique_pid: inverted_index[token].get(unique_pid, 0) + 1,
                }
            )
        except Exception as e:  # noqa: F841
            print(token)
            print(e)
            pass

    return inverted_index, doc_lengths


def get_inverted_index():

    vocab_df_sorted = pd.read_csv(
        "passage_collection_stats_no_stopwords.csv",
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
    nltk.download("stopwords")
    stop_tokens = task1.stopwords_dict()
    inverted_index = get_inverted_index()
    # print(inverted_index)
    unique_pids, unique_passages = get_unique_pids()
    # get all unique 182469 passages and thier ids
    doc_lengths: dict = {}
    for index, [pid, passage] in tqdm(
        enumerate(zip(unique_pids, unique_passages))
    ):
        inverted_index, doc_lengths = work_on_one_passage(
            pid, passage, inverted_index, doc_lengths, stop_tokens
        )

        # if index == 1000:
        #     break

    with open("inverted_index.pickle", "wb") as f:
        pickle.dump(inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving inverted index as .pickle file")

    with open("doc_lengths.pickle", "wb") as f:
        pickle.dump(doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving doc_lengths as .pickle file")
