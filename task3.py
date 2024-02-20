import pickle
import numpy as np

import pandas as pd
from tqdm import tqdm  # type:ignore

import task1
import task2


def get_test_queries() -> tuple[pd.Series, pd.Series]:
    """
    get_test_queries read the test-queries.csv and return queries
    and query ID as series

    _extended_summary_

    :return: 2 series, unique qid, query text
    :rtype: tuple[pd.Series, pd.Series]
    """

    test_queries_df = pd.read_csv(
        "test-queries.tsv",
        header=None,
        delimiter="\t",
    )
    test_queries_df.rename(columns={0: "qid", 1: "query"}, inplace=True)

    return test_queries_df["qid"], test_queries_df["query"]


def get_inverted_index() -> dict:
    """
    get_inverted_index Read the inverted index stored as a pickle file
    add a count key to all tokens and return it

    Adds a count key to all tokens that is a count of all documents
    that token appears in

    :return: Inverted index as a dictionary
    :rtype: dict
    """

    with open("inverted_index.pickle", "rb") as handle:
        inverted_index = pickle.load(handle)

    for token in inverted_index.keys():
        inverted_index[token]["count"] = len(inverted_index[token].keys())
    return inverted_index


def TF_IDF_one_passage(
    pid: int, passage: str, corpus_size: float, inverted_index
) -> dict:
    """
    TF_IDF_one_passage TF-IDF vector for a single passage

    Only store values for tokens present in the passage

    :param pid: unique passage ID
    :type pid: int
    :param passage: passage as a string
    :type passage: str
    :param corpus_size: size of all unique passages
    :type corpus_size: float
    :return: TF-IDF vector as a dictionary
    :rtype: dict
    """

    passage_tokens = task1.work_one_line(passage)
    total_tokens = float(len(passage_tokens))
    TF_IDF_vec = dict.fromkeys(set(passage_tokens))

    for token in TF_IDF_vec:
        TF_IDF_vec[token] = (
            inverted_index[token][pid] / total_tokens
        ) * np.log(corpus_size / inverted_index[token]["count"])

    return TF_IDF_vec


def TF_IDF_all_passages() -> int:
    """
    TF_IDF_all_passages Generate and store a dictionary of TF-IDF of
    all passages

    dictionary stored as a pickle file

    :return: total number of passages aka corpus size
    :rtype: int
    """
    inverted_index = get_inverted_index()

    pids, passages = task2.get_unique_pids()
    corpus_size = pids.shape[0]
    tf_idf_passages = {}
    for pid, passage in tqdm(zip(pids, passages)):
        # print(pid, passage)
        tf_idf_passages[pid] = TF_IDF_one_passage(
            pid, passage, float(corpus_size), inverted_index
        )
        # break
    with open("tf_idf_passages.pickle", "wb") as f:
        pickle.dump(tf_idf_passages, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving TF-IDF of all passages as .pickle file")

    return corpus_size


def TF_IDF_one_query(
    qid: int, query: str, corpus_size: float, inverted_index
) -> dict:
    """
    TF_IDF_one_query generate TF-IDF vector for one query at
    a time

    For IDF we use the inverted index stored before

    :param qid: unique query ID
    :type qid: int
    :param query: query as a string
    :type query: str
    :param corpus_size: Total number of unique passages
    :type corpus_size: float
    :return: TF-IDF vector as a dictionary
    :rtype: dict
    """

    query_tokens = task1.work_one_line(query)
    total_tokens = float(len(query_tokens))
    query_inverted_index: dict = {}

    for token in query_tokens:

        query_inverted_index[token] = query_inverted_index.get(token, 0) + 1
    TF_IDF_vec = dict.fromkeys(set(query_tokens))

    remove_keys = []
    for token in TF_IDF_vec:
        try:
            TF_IDF_vec[token] = (
                query_inverted_index[token] / total_tokens
            ) * np.log(corpus_size / inverted_index[token]["count"])
        except Exception as e:
            print(f"{token} not in inverted index, {e}")
            remove_keys.append(token)

    for k in remove_keys:
        TF_IDF_vec.pop(k, None)

    return TF_IDF_vec


def TF_IDF_all_queries(corpus_size: float):
    """
    TF_IDF_all_queries iterate over all queries and
    calculate and store TF_IDF score

    _extended_summary_

    :param corpus_size: _description_
    :type corpus_size: float
    """

    inverted_index = get_inverted_index()

    qids, queries = get_test_queries()
    tf_idf_queries = {}
    for qid, query in tqdm(zip(qids, queries)):
        tf_idf_queries[qid] = TF_IDF_one_query(
            qid, query, float(corpus_size), inverted_index
        )
        # break

    with open("tf_idf_queries.pickle", "wb") as f:
        pickle.dump(tf_idf_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving TF-IDF of all queries as .pickle file")


def cosine_score(tf_idf_passage, tf_idf_query):

    score = None

    for q_token, q_token_tf_idf in tf_idf_query.items():
        if q_token in tf_idf_passage:

            if score is None:
                score = tf_idf_passage[q_token] * q_token_tf_idf
            else:
                score += tf_idf_passage[q_token] * q_token_tf_idf

    if score is None:
        return float("-inf")
    else:
        return score


def top100_pids_score(qid, tf_idf_passages, tf_idf_query, unique_pids):

    # top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    for index, pid in enumerate(unique_pids):

        scores[index] = cosine_score(tf_idf_passages[pid], tf_idf_query)

    valid_scores = scores[~np.isinf(scores)]
    valid_pids = pids[~np.isinf(scores)]

    sorted_index = np.argsort(valid_scores)
    sorted_scores = valid_scores[sorted_index][::-1]
    sorted_pids = valid_pids[sorted_index][::-1]

    top100_scores = sorted_scores[:100]
    top100_pids = sorted_pids[:100]

    top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])
    top100_pid_score_df["qid"] = qids[: top100_pids.shape[0]]
    top100_pid_score_df["pid"] = top100_pids
    top100_pid_score_df["score"] = top100_scores

    return top100_pid_score_df


def iterate_qids():

    with open("tf_idf_passages.pickle", "rb") as handle:
        tf_idf_passages = pickle.load(handle)

    with open("tf_idf_queries.pickle", "rb") as handle:
        tf_idf_queries = pickle.load(handle)

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

    unique_pids, _ = task2.get_unique_pids()

    qids, _ = get_test_queries()

    df = pd.DataFrame(columns=["qid", "pid", "score"])

    for qid in tqdm(qids):
        candidate_passages_df_qid = candidate_passages_df[
            candidate_passages_df["qid"] == qid
        ]
        qid_unique_pids = candidate_passages_df_qid["pid"]

        qid_df = top100_pids_score(
            qid,
            tf_idf_passages,
            tf_idf_queries[qid],
            qid_unique_pids,
        )
        df = pd.concat([df, qid_df])
        # break

    df.to_csv("tfidf.csv", index=False, header=False)  # add header= False
    print("Done Saving top 100 scores of all queries as .csv file")


corpus_size = TF_IDF_all_passages()
TF_IDF_all_queries(corpus_size)

iterate_qids()
