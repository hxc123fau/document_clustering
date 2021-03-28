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


# text2vec methods
class text2vec():
    def __init__(self, doc_list):
        # Initialize
        self.doc_list = doc_list
        self.nlp, self.docs, self.docs_dict = self._preprocess(self.doc_list)


    # Gensim to create a dictionary and filter out too frequent and infrequent words
    def _get_docs_dict(self, docs):
        docs_dict = Dictionary(docs)
        # CAREFUL: For small corpus
        docs_dict.filter_extremes(no_below=5, no_above=0.2)
        # docs_dict.filter_extremes(no_below=5)
        # after some tokens have been removed remove the gaps
        docs_dict.compactify()
        print('docs_dict',docs_dict)
        return docs_dict

    def _lemmatize_doc(self, doc):
        lemma_doc=[]
        for t in doc:
            # lemma_doc.append(t.lemma_)
            if t.is_alpha and not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num) is True:
                lemma_doc.append(t.lemma_)
        return lemma_doc


    def _preprocess(self, doc_list):
        # nlp = spacy.load('en_core_web_sm')
        nlp = spacy.load('en_core_web_lg')
        # lemmatisation word
        lemma_docs = [self._lemmatize_doc(nlp(doc)) for doc in doc_list]
        docs_dict = self._get_docs_dict(lemma_docs)
        # print('lemma_docs',len(lemma_docs),lemma_docs)
        return nlp, lemma_docs, docs_dict


    def _get_tfidf(self):
        # Convert document (a list of words) into the bag-of-words format
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
        model_tf_idf = TfidfModel(docs_corpus, id2word=self.docs_dict)
        docs_tf_idf = model_tf_idf[docs_corpus]

        docs_tuples=[]
        for c in docs_tf_idf:
            docs_tuples.append(sparse2full(c, len(self.docs_dict)))
            # print('ccc',len(c),c)
        tf_idf_vec=np.vstack(docs_tuples)
        return tf_idf_vec

    def tf_idf_weighted_wv(self):
        # tf-idf
        tf_idf_vec = self._get_tfidf()

        # model = Word2Vec(size=200)
        train_docs=[]
        for doc in self.docs:
            train_docs.extend(doc)

        # print('self.docs',train_docs)
        # corpus = api.load('text8')
        word2vec_model = Word2Vec(self.docs,size=300,min_count=1)
        # model.train(train_docs,total_examples = model.corpus_count,epochs = model.iter)

        word_vec_list=[]
        for i in range(len(self.docs_dict)):
            # print('self.docs_dict[i]',self.docs_dict[i])
            # load glove embedding vector
            # word_vec=self.nlp(self.docs_dict[i]).vector
            word_vec=word2vec_model[self.docs_dict[i]]
            # print('word_vec',word_vec.shape)
            word_vec_list.append(word_vec)
        all_word_vec=np.vstack(word_vec_list)
        weighted_wv = np.dot(tf_idf_vec, all_word_vec)
        # print('weighted_wv',weighted_wv.shape)

        # return tf_idf_vec
        # return all_word_vec
        return weighted_wv

    def doc_vec(self):
        tagged_data=[]
        for i,doc in enumerate(self.docs) :
            tagged_data.append(TaggedDocument(words=doc,tags=[str(i)]))
            # print('tagged_data',tagged_data)

        max_epochs = 100
        vec_size = 100
        alpha = 0.025
        model = Doc2Vec(size=vec_size,alpha=alpha,min_alpha=0.00025,min_count=1,dm=1)
        model.build_vocab(tagged_data)
        for epoch in range(max_epochs):
            # print('iteration {0}'.format(epoch))
            model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        return model.docvecs.doctag_syn0




