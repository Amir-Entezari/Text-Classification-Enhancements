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


def apply_lsa(X_train, X_test, n_components=100):
    """
    Apply Latent Semantic Analysis (LSA) to the document embeddings.

    Parameters:
    - X_train (list of numpy.ndarray): Document embeddings of the training set.
    - X_test (list of numpy.ndarray): Document embeddings of the test set.
    - n_components (int): Number of components to keep in the truncated SVD.

    Returns:
    - tuple: Tuple containing the transformed training and test sets after applying LSA.
    """
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_lsa = lsa.fit_transform(X_train)
    X_test_lsa = lsa.transform(X_test)
    return X_train_lsa, X_test_lsa
