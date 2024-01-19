import os
import re
import string

import pandas as pd
import requests
import tarfile
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_train_test_imdb_data(data_dir):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.

    Returns:
    train/test datasets as pandas dataframes.
    """

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                ID = f_name.split('_')[0]
                score = f_name.split('_')[1].split('.')[0]

                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    data[split].append([ID, review, int(score)])

    # np.random.shuffle(data["train"])
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['ID', 'text', 'rating'])
    # np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['ID', 'text', 'rating'])
    return data["train"], data["test"]


def preprocess_text(text, remove_stopwords=True, stemming=True, lemmatization=True):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert to lower case
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        tokens = [token for token in tokens if token not in stop_words and token.isalpha()]

    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


if __name__ == '__main__':
    DATA_DIR = "../dataset/raw/aclImdb_v1/aclImdb"
    if not os.path.exists(DATA_DIR):
        response = requests.get("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", stream=True)
        if response.status_code == 200:
            with open("aclImdb_v1.tar.gz", "wb") as file:
                file.write(response.raw.read())
            with tarfile.open("aclImdb_v1.tar.gz") as tar:
                tar.extractall(path="../dataset/raw/")
            os.remove("aclImdb_v1.tar.gz")  # Remove the tar file after extraction
        else:
            raise f"Failed to download the file: {response.status_code}"

    train_data, test_data = load_train_test_imdb_data(data_dir=DATA_DIR)
    print(train_data)
    print(test_data)

    train_pos = train_data[train_data['rating'] > 5]
    train_neg = train_data[train_data['rating'] <= 5]
    test_pos = test_data[test_data['rating'] > 5]
    test_neg = test_data[test_data['rating'] <= 5]
    # save csv files
    train_pos.to_csv('../dataset/preprocessed/train_pos.csv')
    train_neg.to_csv('../dataset/preprocessed/train_neg.csv')
    test_pos.to_csv('../dataset/preprocessed/test_pos.csv')
    test_neg.to_csv('../dataset/preprocessed/test_neg.csv')
