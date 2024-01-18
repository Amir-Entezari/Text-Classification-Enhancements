import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class InvertedIndex:
    """
    In this class, I implement an information retrieval system which can search a query among documents.
    ...
    Attributes:
    -----------
    collection_size: int
        number of documents.
    documents: List
        list of documents in format of Document object.
    posting_list: List[Token]
        list of Term objects. Terms store a string , document's indexes , cf and df
    stop_word: set
        set of stop words to check when tokenizing
    case_sensitive: bool
        a boolean to determine whether we want to distinguish between lowercase and uppercase form.
    tf_idf_matrix: List[List] (matrix)
        The tf-idf matrix that described in the slides.


    Methods
    -------
    Methods defined here:
        __init__(self, documents: List, case_sensitive=False):
            Constructor will set initial attributes like case_sensitivity. NOTE that documents should be read with read_document function after creating our IR system.
            :parameter
            ---------
            case_sensitive: bool
                boolean that determine whether the terms are case-sensetive. For more simplicity I set it to false, but It works when set to True too.(when we need to distinguish between a term in our query that has capital or small letter with other terms.)
            :return
                None

        get_token_index(self, x):
            this function find index of a word in posting list using binary search algorithm.
            :parameter
                x:str
                    the word you want to find its index
            :return
                int: index of the word in posting_list

        create_posting_list(self):
            calling this function, will create posting list of all occurred words cross all documents.
            :parameter
                None
            :return
                None

        create_tf_idf_matrix(self):
            This function will create a tf-idf matrix. I used the formula in the slides.
            :parameter
                None
            :return
                None
    """

    def __init__(self, dataset, case_sensitive=False):
        self.dataset = dataset
        self.posting_list = []
        self.positive_posting_list = []
        self.negative_posting_list = []

        self.tf_idf_matrix = []

    def get_term_index(self, posting_list, term):
        """
        This function find index of a word in posting list using binary search algorithm.
            :parameter
                x:str
                    the word you want to find its index
            :return
                int: index of the word in posting_list
        """
        low = idx = 0
        high = len(posting_list) - 1
        while low <= high:
            if high - low < 2:
                if posting_list[high]['word'] < term:
                    idx = high + 1
                    break
                elif posting_list[high]['word'] == term:
                    idx = high
                    break
                elif posting_list[low]['word'] >= term:
                    idx = low
                    break
            idx = (high + low) // 2
            if posting_list[idx]['word'] < term:
                low = idx + 1
            elif posting_list[idx]['word'] > term:
                high = idx - 1
            else:
                break
        return idx

    def create_posting_list(self, dataset):
        """
        calling this function, will create posting list of all occurred words cross all documents of the given dataset. in this function, I
        loop over all documents, then inside this loop, I loop over all the tokens that are in the current document.
        then I check if the length of posting_list is zero, then I add this token as first term. else if the length of
        posting_list is more than 0, I find the correct index of the token in posting_list alphabetically. then I check
        if this token, has been already in posting_list, I just add the current document index in tokens.docs, else, I
        add this token in the posting_list, then add the current document index. I also calculate cf and df during the loops.
            :parameter
                dataset
            :return
                The posting list of the given dataset
        :return:
        """
        posting_list = []
        for doc_id, doc in dataset.iterrows():
            doc_text = doc['text']
            tokenized_text = word_tokenize(doc_text)
            for token_idx, token in enumerate(tokenized_text):
                if len(posting_list) == 0:
                    posting_list.append({'word': token,
                                         'docs': [],
                                         'df': 0,
                                         'cf': 0,
                                         })
                    posting_list[0]['docs'].append({doc_id: [token_idx]})
                    posting_list[0]['cf'] += 1
                    continue

                idx = self.get_term_index(posting_list, token)

                if idx == len(posting_list):
                    posting_list.append({'word': token,
                                         'docs': [],
                                         'df': 0,
                                         'cf': 0,
                                         })
                    # self.posting_list[i].post_idx.append(post_idx)
                elif token != posting_list[idx]['word']:
                    posting_list.insert(idx, {'word': token,
                                              'docs': [],
                                              'df': 0,
                                              'cf': 0,
                                              })

                if len(posting_list[idx]['docs']) == 0:
                    posting_list[idx]['docs'].append({doc_id: [token_idx]})
                    posting_list[idx]['df'] += 1
                elif doc_id not in posting_list[idx]['docs'][-1].keys():
                    posting_list[idx]['docs'].append({doc_id: [token_idx]})
                    posting_list[idx]['df'] += 1
                else:
                    posting_list[idx]['docs'][-1][doc_id].append(token_idx)
                posting_list[idx]['cf'] += 1
        return posting_list

    def idf(self, df_t):
        return np.log(self.dataset.shape[0] / df_t)

    def create_tf_idf_matrix(self, dataset, use_sklearn=True):
        """
        This function will create a tf-idf matrix. I used the formula in the slides. Fisrt I set all values of the matrix to zeros then I loop over all terms in posting list and then loop over all documents in each term, an set the row of t-th term and doc_idx-th column to tf*idf.
        :return:
            None
        """
        if use_sklearn:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(dataset)
            return X
        else:
            collection_size = self.dataset.shape[0]
            tf_idf_matrix = np.zeros([len(self.posting_list), len(self.dataset)])
            print(tf_idf_matrix.shape)
            for t in range(len(self.posting_list)):
                for doc in self.posting_list[t]['docs']:
                    doc_idx, indexes = next(iter(doc.items()))
                    tf_idf_matrix[t, doc_idx] = len(doc[doc_idx]) * np.log(
                        collection_size / self.posting_list[t]['df'])

            for col_idx in range(len(self.dataset)):
                v_norm = np.linalg.norm(tf_idf_matrix[:, col_idx])
                if v_norm != 0:
                    tf_idf_matrix[:, col_idx] = tf_idf_matrix[:, col_idx] / v_norm
            return tf_idf_matrix

