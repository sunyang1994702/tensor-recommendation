from restaurantResSys.normalization_one import normalize_corpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


review_file = "../file_package/RichmondHill_review.txt"
myfile = open(review_file, 'r')
CORPUS = []
for line_index, line in enumerate(myfile.readlines()):
    line = line.rstrip()
    CORPUS.append(line)

def doc_vec(CORPUS):
    TOKENIZED_CORPUS = normalize_corpus(CORPUS, True)
    vector_size_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,70,80,90,100,150,200]
    for vs in vector_size_list:
        fname = get_tmpfile("doc2vec_model__" + str(vs))
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(TOKENIZED_CORPUS)]
        model = Doc2Vec(documents, vector_size=vs, window=5, min_count=5, workers=4)
        model.save(fname)

if __name__ == '__main__':
    doc_vec(CORPUS)




"""
vector_size_list = [5,10,15,20,25,30,35,40,45,50,55,60]
for vs in vector_size_list:
    fname = get_tmpfile("doc2vec_model_" + str(vs))
    model = Doc2Vec.load(fname)

    print(len(model[0]))
    print(model[0])
"""