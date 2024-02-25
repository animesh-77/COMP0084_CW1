import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm  # type:ignore

import task1
import task3


def score_Laplace(
    query: str,
    pid: int,
    inverted_index: dict,
    vocab_length: int,
    doc_length: int,
) -> float:
    """
    score_Laplace get query max Likelihood score with
    Laplace smoothing

    _

    :param query: string of the query
    :type query: str
    :param pid: unique passage ID
    :type pid: int
    :param inverted_index: inverted index of whole vocabulary
    :type inverted_index: dict
    :param vocab_length: number of words in whole vocabulary
    :type vocab_length: int
    :param doc_length: Total number of words in the passage after stemming
    :type doc_length: int
    :return: Maximum likelihood estimate of the query from this passage
    :rtype: float
    """

    query_tokens = set(task1.work_one_line(query))

    score = 0.0

    for query_token in query_tokens:
        num = inverted_index.get(query_token, {}).get(pid, 0) + 1
        den = float(doc_length) + float(vocab_length)

        score += np.log(num) - np.log(den)

    return score


def top100_pids_score_Laplace(
    qid: int,
    query: str,
    unique_pids: pd.Series,
    inverted_index: dict,
    vocab_length: int,
) -> pd.DataFrame:
    """
    top100_pids_score_Laplace Get at most the top 100 passages with highest
    ML score for the specific query from the given passages and PIDs

    While calculating the ML for query, Laplace smoothing is applied

    :param qid: unique query ID
    :type qid: int
    :param query: suery string
    :type query: str
    :param unique_pids: List of all unique PIDS to iterate through
    :type unique_pids: pd.Series
    :param inverted_index: inverted index of whole vocabulary
    :type inverted_index: dict
    :param vocab_length: total number of words in the vocabulary
    :type vocab_length: int
    :return: dataframe with at most top 100 passages for this query
    :rtype: pd.DataFrame
    """

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    with open("doc_lengths.pickle", "rb") as handle:
        doc_lengths = pickle.load(handle)

    for index, pid in enumerate(unique_pids):

        scores[index] = score_Laplace(
            query,
            pid,
            inverted_index,
            vocab_length,
            doc_lengths[pid],
        )

    sorted_index = np.argsort(scores)
    sorted_scores = scores[sorted_index][::-1]
    sorted_pids = pids[sorted_index][::-1]

    top100_scores = sorted_scores[:100]
    top100_pids = sorted_pids[:100]

    top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])
    top100_pid_score_df["qid"] = qids[: top100_pids.shape[0]]
    top100_pid_score_df["pid"] = top100_pids
    top100_pid_score_df["score"] = top100_scores

    return top100_pid_score_df


def Laplace_smoothing():
    """
    Laplace_smoothing Iterate over all qids, get at most top 100 pids
    and save in  a .csv file

    Laplace smoothing applied in query likelihood estimation
    """

    inverted_index = task3.get_inverted_index()
    vocab_length = len(inverted_index.keys())

    total_words = get_total_words()

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

    qids, queries = task3.get_test_queries()

    df_laplace = pd.DataFrame(columns=["qid", "pid", "score"])
    df_lidstone = pd.DataFrame(columns=["qid", "pid", "score"])
    df_dirichlet = pd.DataFrame(columns=["qid", "pid", "score"])

    for qid, query in tqdm(zip(qids, queries)):
        candidate_passages_df_qid = candidate_passages_df[
            candidate_passages_df["qid"] == qid
        ]
        qid_unique_pids = candidate_passages_df_qid["pid"]
        # work only with these pids and passages

        qid_df_laplace = top100_pids_score_Laplace(
            qid,
            query,
            qid_unique_pids,
            inverted_index,
            vocab_length,
        )
        df_laplace = pd.concat([df_laplace, qid_df_laplace])

        qid_df_lidstone = top100_pids_score_Lidstone(
            qid,
            query,
            qid_unique_pids,
            inverted_index,
            vocab_length,
        )
        df_lidstone = pd.concat([df_lidstone, qid_df_lidstone])

        qid_df_dirichlet = top100_pids_score_Dirichlet(
            qid,
            query,
            qid_unique_pids,
            inverted_index,
            total_words,
        )
        df_dirichlet = pd.concat([df_dirichlet, qid_df_dirichlet])
        # break
        # break

    df_laplace.to_csv(
        "laplace.csv", index=False, header=False
    )  # add header= False
    print("Done Saving top 100 Laplace scores of all queries as .csv file")

    df_lidstone.to_csv(
        "lidstone.csv", index=False, header=False
    )  # add header= False
    print("Done Saving top 100 Lidstone scores of all queries as .csv file")

    df_dirichlet.to_csv(
        "dirichlet_imp.csv", index=False, header=False
    )  # add header= False
    print("Done Saving top 100 Dirichlet scores of all queries as .csv file")


def score_Lidstone(
    query: str,
    pid: int,
    inverted_index: dict,
    vocab_length: int,
    doc_length: int,
    epsilon: float = 0.1,
) -> float:
    """
    score_Lidstone get Maximum likelihood estimate for the
    given query and passage

    Uses Lidstone smoothing while estimating ML for the query

    :param query: query as a string
    :type query: str
    :param pid: unique passage ID
    :type pid: int
    :param inverted_index: inverted index for the whole vocabulary
    :type inverted_index: dict
    :param vocab_length: total number of words in the vocabulary
    :type vocab_length: int
    :param doc_length: total number of words in given passage/document
    :type doc_length: int
    :param epsilon: parameter for Lidstone correction, defaults to 0.1
    :type epsilon: float, optional
    :return: Maximum likelihood estimate for the given query
    :rtype: float
    """

    query_tokens = set(task1.work_one_line(query))

    score = 0.0

    for query_token in query_tokens:
        num = inverted_index.get(query_token, {}).get(pid, 0) + epsilon
        den = float(doc_length) + epsilon * float(vocab_length)

        score += np.log(num) - np.log(den)

    return score


def top100_pids_score_Lidstone(
    qid: int,
    query: str,
    unique_pids: pd.Series,
    inverted_index: dict,
    vocab_length: int,
) -> pd.DataFrame:
    """
    top100_pids_score_Lidstone For a given query
    get at most the top 100 passages with highest query likelihdod

    While estimating the ML score for a query Lidstone smoothing is applied

    :param qid: unique query ID
    :type qid: int
    :param query: suery string
    :type query: str
    :param unique_pids: List of all unique PIDS to iterate through
    :type unique_pids: pd.Series
    :param inverted_index: inverted index of whole vocabulary
    :type inverted_index: dict
    :param vocab_length: total number of words in the vocabulary
    :type vocab_length: int
    :return: dataframe with at most top 100 passages for this query
    :rtype: pd.DataFrame
    """

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    with open("doc_lengths.pickle", "rb") as handle:
        doc_lengths = pickle.load(handle)

    for index, pid in enumerate(unique_pids):

        scores[index] = score_Lidstone(
            query,
            pid,
            inverted_index,
            vocab_length,
            doc_lengths[pid],
        )

    sorted_index = np.argsort(scores)
    sorted_scores = scores[sorted_index][::-1]
    sorted_pids = pids[sorted_index][::-1]

    top100_scores = sorted_scores[:100]
    top100_pids = sorted_pids[:100]

    top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])
    top100_pid_score_df["qid"] = qids[: top100_pids.shape[0]]
    top100_pid_score_df["pid"] = top100_pids
    top100_pid_score_df["score"] = top100_scores

    return top100_pid_score_df


def score_Dirichlet(
    query: str,
    pid: int,
    inverted_index: dict,
    doc_length: int,
    mu: float = 50.0,
    total_words: int = 10239641,
) -> float:

    query_tokens = set(task1.work_one_line(query))

    score = 0.0

    for query_token in query_tokens:
        A1 = (doc_length * inverted_index.get(query_token, {}).get(pid, 0)) / (
            (doc_length + mu) * doc_length
        )

        A2 = (mu * inverted_index.get(query_token, {}).get("count", 0)) / (
            (doc_length + mu) * total_words
        )

        B = A1 + A2

        try:
            if B == 0:
                raise ZeroDivisionError
        except ZeroDivisionError:
            print(query_token)
            continue

        score += np.log(B)

    return score


def top100_pids_score_Dirichlet(
    qid: int,
    query: str,
    unique_pids: pd.Series,
    inverted_index: dict,
    total_words: int = 10239641,
) -> pd.DataFrame:

    with open("doc_lengths.pickle", "rb") as handle:
        doc_lengths = pickle.load(handle)

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    for index, pid in enumerate(unique_pids):

        scores[index] = score_Dirichlet(
            query,
            pid,
            inverted_index,
            doc_lengths[pid],
            50.0,
            total_words,
        )

    sorted_index = np.argsort(scores)
    sorted_scores = scores[sorted_index][::-1]
    sorted_pids = pids[sorted_index][::-1]

    top100_scores = sorted_scores[:100]
    top100_pids = sorted_pids[:100]

    top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])
    top100_pid_score_df["qid"] = qids[: top100_pids.shape[0]]
    top100_pid_score_df["pid"] = top100_pids
    top100_pid_score_df["score"] = top100_scores

    return top100_pid_score_df


def get_total_words() -> int:
    with open("doc_lengths.pickle", "rb") as handle:
        doc_lengths = pickle.load(handle)

    total_words = 0

    for doc in tqdm(doc_lengths):
        total_words += doc_lengths[doc]

    print(f"Total number of words is {total_words}")
    return total_words


if __name__ == "__main__":
    Laplace_smoothing()
