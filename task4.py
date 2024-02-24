import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm  # type:ignore

import task1
import task3


def score_laplace(
    query: str,
    pid: int,
    inverted_index: dict,
    vocab_length: int,
    doc_length: int,
) -> float:

    query_tokens = set(task1.work_one_line(query))

    score = 1.0

    for query_token in query_tokens:
        num = inverted_index.get(query_token, {}).get(pid, 0) + 1
        den = float(doc_length) + float(vocab_length)

        score *= num / den

    return score


def top100_pids_score_Laplace(
    qid: int,
    query: str,
    unique_pids: pd.Series,
    inverted_index: dict,
    vocab_length: int,
) -> pd.DataFrame:

    qids = np.ones(unique_pids.shape[0], dtype=int) * qid
    scores = np.zeros(unique_pids.shape[0])
    pids = unique_pids.to_numpy()

    with open("doc_lengths.pickle", "rb") as handle:
        doc_lengths = pickle.load(handle)

    for index, pid in enumerate(unique_pids):

        scores[index] = score_laplace(
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

    inverted_index = task3.get_inverted_index()
    vocab_length = len(inverted_index.keys())

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

    df = pd.DataFrame(columns=["qid", "pid", "score"])

    for qid, query in tqdm(zip(qids, queries)):
        candidate_passages_df_qid = candidate_passages_df[
            candidate_passages_df["qid"] == qid
        ]
        qid_unique_pids = candidate_passages_df_qid["pid"]
        # work only with these pids and passages

        qid_df = top100_pids_score_Laplace(
            qid,
            query,
            qid_unique_pids,
            inverted_index,
            vocab_length,
        )
        df = pd.concat([df, qid_df])
        # break

    df.to_csv("laplace.csv", index=False, header=False)  # add header= False
    print("Done Saving top 100 scores of all queries as .csv file")


if __name__ == "__main__":
    Laplace_smoothing()
