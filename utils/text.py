from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import RSLPStemmer

tokenizer = RegexpTokenizer(r'\w+')

def normalize(doc):
    return unidecode(doc)

def remove_stopwords(doc):
    myStops = ["ta", "ai", "ne", "ia"]
    stop = stopwords.words('portuguese') + myStops
    return ' '.join([word for word in tokenizer.tokenize(doc) if word not in stop and not word.isdigit()])

def apply_stemmer(doc):
    stemmer = RSLPStemmer()
    return ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(doc)])

def clean(doc):
    doc = doc.lower()
    doc = normalize(doc)  # remove acentuacao
    doc = remove_stopwords(doc)
    #doc = apply_stemmer(doc)
    return doc
