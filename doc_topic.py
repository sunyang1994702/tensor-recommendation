from normalization_one import normalize_corpus
from sklearn.decomposition import LatentDirichletAllocation
from restaurantResSys.LDA.utils import build_feature_matrix
import numpy as np
from operator import itemgetter


"""
using scikit-learn to generate LDA model

"""

def train_LDA_model_scikitLearn(corpus, total_topics=2):
    #get tf-idf based features
    norm_corpus = normalize_corpus(corpus)
    vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, feature_type='tfidf')

    #build LDA model
    lda = LatentDirichletAllocation(n_components=total_topics, max_iter=500, learning_method='online', learning_offset=50., random_state=0)
    #get document-topic distribution
    doc_topics = lda.fit_transform(tfidf_matrix)

    #get topic-word distribution
    feature_names = vectorizer.get_feature_names()
    weights = lda.components_


    return doc_topics, feature_names, weights

def print_LDA_topic_words(total_topics, feature_names, weights, display_weights=False, num_term=None):
    for topic_index in range(total_topics):
        topic_words = []
        for w_index, word in enumerate(feature_names):
            topic_words.append((word, weights[topic_index][w_index]))

        topic_words = sorted(topic_words, key=itemgetter(1), reverse=True)

        if display_weights:
            print('Topic #' + str(topic_index+1) + 'with weights')
            print(topic_words[:num_term] if num_term else topic_words)

        else:
            print('Topic #' + str(topic_index + 1) + 'without weights')
            topic_words_without_weight = [each_word for each_word, word_weight in topic_words]
            print(topic_words_without_weight[:num_term] if num_term else topic_words_without_weight)
        print()

def print_LDA_doc_topics(doc_topics, num_doc=None):
    for doc_index, doc in enumerate(doc_topics[:num_doc] if num_doc else doc_topics):
        print('doc # ' + str(doc_index+1))
        print(doc)
        print()

def save_LDA_topic_words(file_topic_words, total_topics, feature_names, weights, num_term=None):
    myfile = open(file_topic_words, 'w+')
    for topic_index in range(total_topics):
        topic_words = []
        for w_index, word in enumerate(feature_names):
            topic_words.append((word, weights[topic_index][w_index]))

        topic_words = sorted(topic_words, key=itemgetter(1), reverse=True)[:num_term]

        for each_word, word_weight in topic_words:
            myfile.write(str(each_word) + ',')
        myfile.write('\n')

    myfile.close()
    
def save_LDA_doc_topics(file_doc_topics, doc_topics):
    myfile = open(file_doc_topics, 'w+')
    for doc_index, topics in enumerate(doc_topics):
        for topic in topics:
            myfile.write(str(topic) + ',')
        myfile.write('\n')

    myfile.close()



def doc_topic_distribution(train, total_topics):
    review_file = "file_package/RichmondHill_review.txt"
    myfile = open(review_file, 'r')
    review_dic = {}
    for line_index, line in enumerate(myfile.readlines()):
        line = line.rstrip()
        review_dic[line_index] = line
    CORPUS = []
    review_index_list = []
    for t_tuple in train:
        review_index = int(t_tuple[3])
        review_index_list.append(review_index)
        CORPUS.append(review_dic[review_index])
    doc_topic, feature_names, weights = train_LDA_model_scikitLearn(CORPUS, total_topics)
    doc_topics = {}
    re_num = 0
    for r_i in sorted(review_index_list):
        doc_topics[r_i] = doc_topic[re_num]
        re_num += 1

    return doc_topics
