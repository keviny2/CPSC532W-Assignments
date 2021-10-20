from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from joint_log_lik import joint_log_lik
from sample_topic_assignment import sample_topic_assignment

bagofwords = loadmat('../data/bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  # go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('../data/words_nips.mat')['WO'][:, 0]
titles = loadmat('../data/titles_nips.mat')['titles'][:, 0]

# This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

# subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
# PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
# words = words[document_assignment < 100]
# document_assignment = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

# number of topics
n_topics = 20

# initial topic assignments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

# within document count of topics
# N_kj = N_td = # times a word in document d is assigned to topic t
doc_counts = np.zeros((n_docs, n_topics))

for d in range(n_docs):
    # histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = \
        np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5, n_topics - 0.5))[0]

# doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

# within topic count of words
# N_wk = N_nt = # times a word n is assigned to topic t
topic_counts = np.zeros((n_topics, alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5, alphabet_size - 0.5))[0]

# topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

# prior parameters, alpha parameterizes the dirichlet to regularize the
# document specific distributions over topics and gamma parameterizes the
# dirichlet to regularize the topic specific distributions over words.
# These parameters are both scalars and really we use alpha * ones() to
# parameterize each dirichlet distribution. Iters will set the number of
# times your sampler will iterate.


# tried:
# alpha=0.1, gamma=0.1
# alpha=0.1, gamma=0.001
# alpha=2, gamma=2
alpha = 0.1
gamma = 0.001
num_burnin = 1000
iters = 10000
# https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf

jll = []
for i in range(iters):
    print("=" * 5, "Iteration:", i, "=" * 5)
    jll.append(joint_log_lik(doc_counts, topic_counts, alpha, gamma))

    prm = np.random.permutation(words.shape[0])

    # mix up data
    words = words[prm]
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]


    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
        topic_assignment,
        topic_counts,
        doc_counts,
        topic_N,
        doc_N,
        alpha,
        gamma,
        words,
        document_assignment)


jll.append(joint_log_lik(doc_counts, topic_counts, alpha, gamma))
del jll[:num_burnin]

plt.plot(jll)
plt.savefig('../results/jll_plot.png')

### find the 10 most probable words of the 20 topics:

# distribution of words in each topic
topic_word_probabilities = (topic_counts + gamma) / (topic_N + alphabet_size * gamma)[:, np.newaxis]

find_top_10 = lambda arr: np.argpartition(arr, -10)[-10:]
most_probable_words_per_topic = np.apply_along_axis(find_top_10, 1, topic_word_probabilities)

convert_int_to_word = lambda arr: WO[arr]
most_probable_words_per_topic = np.apply_along_axis(convert_int_to_word, 1, most_probable_words_per_topic)

np.savetxt('../results/most_probable_words_per_topic.txt', most_probable_words_per_topic, delimiter=" ", fmt="%s")

# fstr = ''
# with open('most_probable_words_per_topic', 'w') as f:
#    f.write(fstr)

# most similar documents to document 0 by cosine similarity over topic distribution:
# normalize topics per document and dot product:
topic_document_probabilities = (doc_counts + alpha) / (doc_N + n_topics * alpha)[:, np.newaxis]

connectivity_versus_entropy = topic_document_probabilities[0]
find_similarity = lambda arr: np.dot(connectivity_versus_entropy, arr)\
                              / (np.linalg.norm(connectivity_versus_entropy)*np.linalg.norm(arr))
similarities = np.apply_along_axis(find_similarity, 1, topic_document_probabilities)
similarities[0] = 0

most_similar_documents = titles[find_top_10(similarities)]
np.savetxt('../results/most_similar_titles_to_0.txt', most_similar_documents, delimiter=' ', fmt="%s")

# fstr = ''
# with open('most_similar_titles_to_0', 'w') as f:
#     f.write(fstr)


