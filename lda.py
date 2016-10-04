__author__ = 'Atanas'

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim



def data_preparation(data, normalization_type="stemming"):

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')

    words = []

    for doc in data:
        doc = doc.lower()
        #Tokenize: Get words from raw data
        tokens = tokenizer.tokenize(doc)
        #Remove stop words
        cleaned_tokens = [i for i in tokens if not i in en_stop]

        if (normalization_type == "stemming"):
            words.append(stemming(cleaned_tokens))
        else:
            words.append(lemmatization(cleaned_tokens))

    return words

def stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(i) for i in tokens]
    return stemmed_tokens

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in tokens]
    return lemmatized_tokens

def lda_process(data, num_topics, passes, num_words, normalization_type):
    words = data_preparation(data, normalization_type)
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(i) for i in words]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=passes)
    list = ldamodel.show_topics(num_topics=num_topics, num_words=num_words)
    return list