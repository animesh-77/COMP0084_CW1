import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # type:ignore
from nltk.stem import PorterStemmer  # type:ignore
from nltk.tokenize import word_tokenize  # type:ignore
from tqdm import tqdm  # type:ignore

all_tokens = {}


def work_one_line(line):

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


def add_tokens(stemmed_tokens):
    for token in stemmed_tokens:
        if token not in all_tokens:
            all_tokens[token] = 1
        else:
            all_tokens[token] += 1


# print(
#     work_one_line(
#         """A@This is a test sentence! It contains special characters:
#             !\"#\$%&'(or)*+does,it-./:;<=>?@[\]^_`{|}~,"""
#     )
# )

with open("passage-collection.txt", "r") as f:
    for line_num, line in enumerate(tqdm(f)):
        # print(line_num, line)
        stemmed_tokens = work_one_line(line)
        add_tokens(stemmed_tokens)
        # if line_num == 500:

        #     break

df = pd.DataFrame.from_dict(all_tokens, orient="index", columns=["Count"])
df = df.reset_index()
df.rename({"index": "1-gram"}, inplace=True)
df.to_csv("passage_collection_stat.csv", index=False)


df = pd.read_csv("passage_collection_stat.csv")

df_sorted = df.sort_values(by=["Count"], ascending=False)
df_sorted["rank"] = (df.index + 1).astype(float)

total_voc = np.sum(df_sorted["Count"])
df_sorted["Freq"] = df_sorted["Count"] / total_voc
df_sorted["Freq*rank"] = df_sorted["Freq"] * df_sorted["rank"]


# print(df_sorted.head(20))


Hn = np.sum(np.reciprocal(df_sorted["rank"]))

zipf_freq = np.reciprocal(df_sorted["rank"] * Hn)


plt.title("Zipf's Law Comparison")
plt.xlabel("Frequence ranking of term")
plt.ylabel("Prob of occurence of term")
plt.plot(df_sorted["rank"], df_sorted["Freq"], color="blue", label="Data")
plt.plot(
    df_sorted["rank"],
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
plt.loglog(df_sorted["rank"], df_sorted["Freq"], color="blue", label="Data")
plt.loglog(
    df_sorted["rank"],
    zipf_freq,
    linestyle="--",
    color="black",
    label="Zipf's curve",
)
plt.grid(True, which="major", ls="-")
plt.legend()
plt.savefig("Figure_2.svg")
plt.show()
