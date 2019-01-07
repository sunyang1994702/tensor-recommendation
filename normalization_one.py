import nltk
from restaurantResSys.replacers import RegexpReplacer, RepeatReplacer
from pattern.text.en import tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import string
from nltk.corpus import stopwords
import time

wnl = WordNetLemmatizer()
stop_words = stopwords.words('english')

"""
runtime :2.4270501136779785
:param text: 1，先用正则表达式替换每个句子中的单词 eg, it's--it is
            2，去掉分好词的数组中的标点符号
            3，去掉停顿词
            4，处理重复字符的单词 eg, happyyyyy--happy
            5, 单词还原
"""
def normalize_corpus(corpus, tokensize=False):
    normalized_corpus = []
    t1 = time.time()
    for text in corpus:
        text = expand_contractions(text)
        text = remove_special_charaters(text)
        text = remove_stopwords(text)
        text = repeated_word(text)
        text = lemmatize_text(text)
        if tokensize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
    t2 = time.time()
    #print(t2-t1)
    return normalized_corpus



#构建一个分词函数
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

#构建扩展缩写词的函数
def expand_contractions(text):
    replacer_regex = RegexpReplacer()
    expanded_text = replacer_regex.replace(text)
    return expanded_text

#构建一个处理重复字符的单词函数
def repeated_word(text):
    tokens = tokenize_text(text)
    replacer_repeat = RepeatReplacer()
    restored_words = [replacer_repeat.replace(token) for token in tokens]
    restored_list = ' '.join(restored_words)

    return restored_list


#单词还原
def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    return tagged_lower_text
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_words = [wnl.lemmatize(word, pos=pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]

    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

#构建一个处理标点符号和字符的函数
def remove_special_charaters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

#构建一个处理停顿词的函数
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_stopwords = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_stopwords)

    return filtered_text





if __name__ == '__main__':
    toy_corpus = [
        "The fox jumps over the dog",
        "The dog is slow and lazy",
        "The dog is very clever and quick",
        "The cat is smarter than the fox and the dog",
        "Python is an excellent programming language",
        "Java and Ruby are other programming languages",
        "Python and Java are very popular programming languages",
        "Python programs are smaller than Java programs"
    ]
    text_array = ["It's a pleasant %9.6 evening.", "Guest, who came ###from US arrived at the venue", "Food was @@@tastyyyy."]

    print(normalize_corpus(toy_corpus, True))
    print(clean_text1(toy_corpus))
