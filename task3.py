import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance  # type:ignore

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
    calculate and store TF_IDF score in a dictionary

    afterwards this dictionary is saved as a pickle for fast
    retrievel

    :param corpus_size: _description_
    :type corpus_size: float
    """

    inverted_index = get_inverted_index()

    qids, queries = get_test_queries()
    tf_idf_queries = {}
    for qid, query in tqdm(zip(qids, queries)):
        tf_idf_queries[qid] = TF_IDF_one_query(
            qid,
            query,
            float(corpus_size),
            inverted_index,
        )
        # break

    with open("tf_idf_queries.pickle", "wb") as f:
        pickle.dump(tf_idf_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done Saving TF-IDF of all queries as .pickle file")


def cosine_score(tf_idf_passage: dict, tf_idf_query: dict) -> float:
    """
    cosine_score Computes the cosine score between 2 TF-IDF vectors,
    one of document and one of query

    scipy is used to generate cosine score between 2 vectors

    :param tf_idf_passage: TF-IDF vector of passage as a dictionary
    :type tf_idf_passage: dict
    :param tf_idf_query: TF-IDF vector of query as a dictionary
    :type tf_idf_query: dict
    :return: cosine similarity of 2 TF-IDF vectors
    :rtype: float
    """

    df_passage = pd.DataFrame.from_dict(
        tf_idf_passage,
        orient="index",
        columns=["passage"],
    )
    df_query = pd.DataFrame.from_dict(
        tf_idf_query,
        orient="index",
        columns=["query"],
    )

    df = pd.merge(
        left=df_passage,
        left_index=True,
        right=df_query,
        right_index=True,
        how="outer",
    )
    df.fillna(value=0.0, inplace=True)

    score = 1 - distance.cosine(df["passage"], df["query"])

    return score


def top100_pids_score(
    qid: int, tf_idf_passages: dict, tf_idf_query: dict, unique_pids: pd.Series
) -> pd.DataFrame:
    """
    top100_pids_score Get top 100 documents based on cosine score between
    TF-IDF vector
    representation of document and query

    Returns a dataframe for this ONE query.
    qid, pid, score

    :param qid: unique query identifier currently being worked on
    :type qid: int
    :param tf_idf_passages: TF-IDF representation of all passasges
    :type tf_idf_passages: dict
    :param tf_idf_query: TF-IDF representation of ONE query
    :type tf_idf_query: dict
    :param unique_pids: list of all documents which are to be ranked
    :type unique_pids: pd.Series
    :return: Dataframe of qid, pid, score .total size <=100
    :rtype: pd.DataFrame
    """

    # top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    for index, pid in enumerate(unique_pids):

        scores[index] = cosine_score(tf_idf_passages[pid], tf_idf_query)

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


def iterate_qids():
    """
    iterate_qids Iterate over all qids get (at most) top 100 documents and
    save final dataframe as a .csv

    Pseude Code
    1) Iterate over all qids
    2) Select candidate passages that are to be re-ranked from
    candidate-passages-top1000.tsv
    3) get (at most) top 100 documents and their scores
    4) store the values in a dataframe and then save it
    """

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


def BM25_score(
    query: str,
    query_inverted_index: dict,
    inverted_index: dict,
    pid: int,
    doc_length: int,
    avg_doc_length: float,
    corpus_size: float,
    k1: float = 1.2,
    k2: float = 100,
    b: float = 0.75,
) -> float:

    K = k1 * ((1 - b) + b * (float(doc_length) / avg_doc_length))

    query_tokens = set(task1.work_one_line(query))
    score = 0.0

    for query_token in query_tokens:
        try:
            a1 = 1 / (corpus_size - inverted_index[query_token]["count"] + 0.5)
            a2 = ((k1 + 1) * inverted_index[query_token][pid]) / (
                K + inverted_index[query_token][pid]
            )
            a3 = ((k2 + 1) * query_inverted_index[query_token]) / (
                k2 * query_inverted_index[query_token]
            )

            score += np.log(a1) * a2 * a3

        except KeyError:
            # print(f"Exception {e} for token {query_token}")
            score += 0

    return score


def top100_pids_score_BM25(
    qid: int,
    query: str,
    qid_unique_pids: pd.Series,
    qid_unique_passages: pd.Series,
    inverted_index: dict,
    avg_doc_length: float,
    doc_lengths: dict,
) -> pd.DataFrame:

    query_inverted_index: dict = {}

    query_tokens = task1.work_one_line(query)

    for query_token in query_tokens:
        query_inverted_index[query_token] = (
            query_inverted_index.get(query_token, 0) + 1
        )

    # query_inverted_index inverted index for current query

    corpus_size = len(doc_lengths.keys())

    qids = np.ones(qid_unique_pids.shape[0], dtype=int) * qid
    scores_BM25 = np.zeros(qid_unique_pids.shape[0])
    pids = qid_unique_pids.to_numpy()

    for index, pid in enumerate(qid_unique_pids):

        scores_BM25[index] = BM25_score(
            query,
            query_inverted_index,
            inverted_index,
            pid,
            doc_lengths[pid],
            avg_doc_length,
            corpus_size,
        )

    sorted_index = np.argsort(scores_BM25)
    sorted_scores_BM25 = scores_BM25[sorted_index][::-1]
    sorted_pids = pids[sorted_index][::-1]

    top100_scores_BM25 = sorted_scores_BM25[:100]
    top100_pids = sorted_pids[:100]

    top100_pid_score_df = pd.DataFrame(columns=["qid", "pid", "score"])
    top100_pid_score_df["qid"] = qids[: top100_pids.shape[0]]
    top100_pid_score_df["pid"] = top100_pids
    top100_pid_score_df["score"] = top100_scores_BM25

    return top100_pid_score_df


def iterate_qids_BM25(corpus_size: int):
    """
    iterate_qids Iterate over all qids get (at most) top 100 documents and
    save final dataframe as a .csv

    Pseude Code
    1) Iterate over all qids
    2) Select candidate passages that are to be re-ranked from
    candidate-passages-top1000.tsv
    3) get (at most) top 100 documents and their scores
    4) store the values in a dataframe and then save it
    """

    inverted_index = get_inverted_index()

    avg_doc_length = 0.0
    doc_lengths: dict = {}

    unique_pid, unique_passage = task2.get_unique_pids()

    try:
        assert unique_pid.shape[0] == corpus_size
    except AssertionError:
        print("Corpus size passed is different than pid-passage pairs")
        corpus_size = unique_pid.shape[0]

    # for pid, passage in tqdm(zip(unique_pid, unique_passage)):

    #     stemmed_tokens = task1.work_one_line(passage)
    #     doc_lengths[pid] = len(stemmed_tokens)
    #     avg_doc_length += doc_lengths[pid]

    # if pid == 8003559:
    #     break

    avg_doc_length = avg_doc_length / float(corpus_size)

    # with open("doc_lengths.pickle", "wb") as f:
    #     pickle.dump(doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print("Done Saving doc_lengths as .pickle file")

    if len(doc_lengths.keys()) == 0:
        with open("doc_lengths.pickle", "rb") as handle:
            doc_lengths = pickle.load(handle)
            avg_doc_length = 56.11715414673177

    # raise ArithmeticError

    print("Corpus Size ", corpus_size)
    print("Size of doc_lengths", len(doc_lengths.keys()))
    print("Average length of document", avg_doc_length)

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

    qids, queries = get_test_queries()

    df = pd.DataFrame(columns=["qid", "pid", "score"])

    for qid, query in tqdm(zip(qids, queries)):

        candidate_passages_df_qid = candidate_passages_df[
            candidate_passages_df["qid"] == qid
        ]
        # for every qid get only the pids in candidate-passages-top1000.csv

        qid_unique_pids = candidate_passages_df_qid["pid"]
        qid_unique_passages = candidate_passages_df_qid["passage"]

        qid_df = top100_pids_score_BM25(
            qid,
            query,
            qid_unique_pids,
            qid_unique_passages,
            inverted_index,
            avg_doc_length,
            doc_lengths,
        )
        df = pd.concat([df, qid_df])
        # break

    df.to_csv("bm25.csv", index=False, header=False)  # add header= False
    print("Done Saving top 100 BM25 scores of all queries as .csv file")


corpus_size = TF_IDF_all_passages()
TF_IDF_all_queries(corpus_size)

iterate_qids()
# corpus_size = 182469
iterate_qids_BM25(corpus_size)
