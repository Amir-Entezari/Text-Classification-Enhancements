import numpy as np
from sklearn.decomposition import TruncatedSVD


def document_embedding(doc, word_vectors):
    """
    Calculate the document embedding by averaging word vectors present in the document.

    Parameters:
    - doc (list of str): Tokenized words in the document.
    - word_vectors (gensim.models.keyedvectors.KeyedVectors): Word vectors for the vocabulary.

    Returns:
    - numpy.ndarray: Document embedding as a numpy array.
    """
    words = [word for word in doc if word in word_vectors]
    if not words:
        return np.zeros(word_vectors.vector_size)
    return np.mean(word_vectors[words], axis=0)


def create_word_embeddings(X, word_vectors):
    """
    Create document embeddings for a given dataset using the provided word vectors.

    Parameters:
    - X (list of list of str): Tokenized documents in the dataset.
    - word_vectors (gensim.models.keyedvectors.KeyedVectors): Word vectors for the vocabulary.

    Returns:
    - list of numpy.ndarray: List of document embeddings.
    """
    return [document_embedding(doc, word_vectors) for doc in X]

