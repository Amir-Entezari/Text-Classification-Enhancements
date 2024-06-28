# Information Retrieval: A Comprehensive Approach with Naive Bayes, Word Embeddings, LSA, and SVM


## Project Overview
This project explores advanced text classification techniques within the scope of information retrieval. By implementing various methods such as TF-IDF, Naive Bayes, Word Embeddings, Latent Semantic Analysis (LSA), and Support Vector Machine (SVM), the study aims to enhance the precision and efficiency of text classification models.

## Key Features
- **Document Preprocessing**: Streamlines text data for better handling in classification tasks.
- **Inverted Index Model**: Utilized for efficient document retrieval.
- **Naive Bayes Classifier**: Implements probabilistic classification with a focus on textual data.
- **Word Embeddings and LSA**: Explores different embedding techniques to capture semantic meanings.
- **SVM Classification**: Applied on both regular and transformed (via LSA) text data to compare performance impacts.

## Installation and Setup
Clone this repository and install the required packages listed in `requirements.txt`:
```
git clone https://github.com/Amir-Entezari/Text-Classification-Enhancements
pip install -r requirements.txt
```
### Dataset Overview
The Large Movie Review Dataset (often referred to as the IMDB dataset) is designed for use in binary sentiment classification, providing a substantial set of 25,000 highly polar movie reviews for training, and 25,000 for testing, making it suitable for developing a benchmark for sentiment analysis. The dataset contains additional unlabeled data for use as well. Each set of reviews is balanced with equal numbers of positive and negative reviews.

### Dataset Citation
- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. *The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)*.

### Downloading and Preparing the Dataset
To download the dataset, use the following link:
[Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

#### Steps to Download and Extract:
1. **Download the dataset** using the link provided above.
2. **Extract the dataset** using a file archiver that supports `.tar.gz` format or you can use the following commands in your terminal:
   ```bash
   wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
   tar -xzf aclImdb_v1.tar.gz

## Usage
To run the project, navigate to the notebook directory and explore the experiment.ipynb script:
```
jupyter notebook experiment.ipynb
```

## Methodology
1. **Document Preprocessing**: Clean and prepare text data.
2. **Naive Bayes Classification**: Train and test using the Bayesian probability model.
3. **Word Embeddings with SVM**: Utilize different embeddings like Word2Vec, GloVe, and FastText with SVM.
4. **LSA with SVM**: Apply dimensionality reduction before SVM classification to analyze impact on performance and training time.

## Results
The project demonstrates that:
- Naive Bayes is highly effective for the targeted text classification tasks.
- Word2Vec provides the best performance among the tested embedding models.
- LSA significantly reduces training time without substantially impacting accuracy.

## Conclusion
Different text classification techniques offer varying levels of efficacy depending on the specific dataset and task requirements. Further research and testing with different configurations and larger datasets are recommended to optimize performance and generalizability.

## Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License
Distributed under the MIT License. See `LICENSE` for more information.