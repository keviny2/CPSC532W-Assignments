import numpy as np
import numba as nb


@nb.jit(nopython=True)
def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.

    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words            @
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1

        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distributions over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    n_words = len(words)
    n_topics = len(topic_N)
    alphabet_size = topic_counts.shape[1]

    topic_assignment_updated = topic_assignment
    topic_counts_updated = topic_counts
    doc_counts_updated = doc_counts
    topic_N_updated = topic_N

    for i in range(n_words):

        # get relevant indices for the current word
        doc_idx = document_assignment[i]
        top_idx = topic_assignment[i]
        word_idx = words[i]

        # remove the current word from counts
        topic_counts_updated[top_idx][word_idx] -= 1
        doc_counts_updated[doc_idx][top_idx] -= 1
        topic_N_updated[top_idx] -= 1


        a = doc_counts_updated[doc_idx] + alpha
        b = (topic_counts_updated[:, word_idx] + gamma) / (topic_N_updated + alphabet_size * gamma)
        probabilities = a * b
        probabilities /= np.sum(probabilities)

        # sample new topic assignment
        new_top_idx = np.argmax(np.random.multinomial(1, probabilities))

        # update counts
        topic_assignment[i] = new_top_idx
        topic_counts_updated[new_top_idx][word_idx] += 1
        doc_counts_updated[doc_idx][new_top_idx] += 1
        topic_N_updated[new_top_idx] += 1

    return topic_assignment_updated, \
           topic_counts_updated, \
           doc_counts_updated, \
           topic_N_updated
