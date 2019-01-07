from restaurantResSys.LDA import LDA_MODEL2



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
    doc_topic, feature_names, weights = LDA_MODEL2.train_LDA_model_scikitLearn(CORPUS, total_topics)
    doc_topics = {}
    re_num = 0
    for r_i in sorted(review_index_list):
        doc_topics[r_i] = doc_topic[re_num]
        re_num += 1

    return doc_topics