import spacy
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models, similarities
from gensim.matutils import sparse2full
import numpy as np
import math
from load_json import get_json_value_by_key
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from  scipy.spatial.distance import euclidean
import scipy
from Clustering_K_Mean import k_mean
import gensim
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from Text2vec import *
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA



def main():
    with open('NLP-test.json', 'rb') as f:
        data = json.loads(f.read())

    doc_list=get_json_value_by_key(data, "fulltext")
    # doc_list = get_json_value_by_key(data, "abstract")
    # doc_list = get_json_value_by_key(data, "headline")

    text2vec_class=text2vec(doc_list)

    cluster_vec=text2vec_class.tf_idf_weighted_wv()   # weighted_word2vec
    # cluster_vec=text2vec_class.doc_vec()  # doc2vec
    # cluster_vec=text2vec_class._get_tfidf() # tf_idf_vec
    # print('cluster_vec',cluster_vec.shape)

    input_title='pca_tf_idf_weighted_word2vec'
    # input_title='doc2vec'
    # input_title='tf_idf_vec'

    pca_model =PCA(n_components=100)
    pca_model.fit(cluster_vec)
    low_dim_embedding = pca_model.transform(cluster_vec)
    # low_dim_embedding=cluster_vec
    print('low_dim_embedding',low_dim_embedding.shape)

    k_mean().elbow_method(low_dim_embedding,input_title,K = range(1, 10))
    k_mean().gap_statistic(low_dim_embedding,input_title,nrefs=20,ks=range(1, 10))
    # k_mean().optimalK(low_dim_embedding, nrefs=20, maxClusters=10)

    # tsne = TSNE(n_components=2)
    # low_dim_embedding=tsne.fit_transform(cluster_vec)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # X, color = datasets.samples_generator.make_s_curve(300, random_state=0)
    # plt.scatter(low_dim_embedding[:, 0], low_dim_embedding[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("2d map")
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.show()


if __name__=="__main__":
    main()