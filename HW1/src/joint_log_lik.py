from scipy.special import loggamma
import numba_scipy
from numba import jit

@jit(nopython=True)
def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model

    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """

    n_docs = doc_counts.shape[0]
    n_topics = doc_counts.shape[1]
    alphabet_size = topic_counts.shape[1]

    ll = 0

    for j in range(n_docs):
        term_1 = 0
        for i in range(n_topics):
            term_1 += loggamma(doc_counts[j][i] + alpha)

        term_2 = 0
        for i in range(n_topics):
            term_2 += doc_counts[j][i] + alpha
        term_2 = loggamma(term_2)
        ll += (term_1 - term_2)

    for i in range(n_topics):
        term_1 = 0
        for r in range(alphabet_size):
            term_1 += loggamma(topic_counts[i][r] + gamma)

        term_2 = 0
        for r in range(alphabet_size):
            term_2 += topic_counts[i][r] + gamma
        term_2 = loggamma(term_2)
        ll += (term_1 - term_2)

    return ll