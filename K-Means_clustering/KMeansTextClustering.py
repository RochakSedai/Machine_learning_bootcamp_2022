import collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def tokenizer(text):
    # transfer the text into array of words
    tokens = word_tokenize(text)
    # yields the stem (fishing-fish, fisher-fish)
    stemmer = PorterStemmer()
    # we filter out stop words
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def cluster_sentence(sentences, nb_of_clusters=2):
    # create tf ifd again: stopwords-> we filter out common words (I,my, the, and...)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'), lowercase=True)
    # builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)

    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)


if __name__ == '__main__':

    sentences = ["Quantuum physics is quite important in science nowadays.",
             "Software engineering is hotter and hotter topic in the silicon valley",
             "Investing in stocks and trading with them are not that easy",
             "FOREX is the stock market for trading currencies",
             "Warren Buffet is famous for making good investments. He knows stock markets"]

    n_clusters = 2
    clusters = cluster_sentence(sentences, n_clusters)

    for cluster in range(n_clusters):
        print("CLUSTER ", cluster, ":")
        for i, sentence in enumerate(clusters[cluster]):
            print("\tSENTENCE ", i, ": ", sentences[sentence])


