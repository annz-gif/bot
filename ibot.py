import numpy as np
import nltk
import string
import random
f = open('/content/chatbot.txt','r',errors = 'ignore')
raw_doc = f.read()
raw_doc =raw_doc.lower() #to convert in lowercase
nltk.download('punkt')# using punkt tokenizer
nltk.download('wordnet')# using wordnet dictionary
nltk.download('omw-1.4')
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
sentence_tokens[:2]
lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(tokens) for token in tokens]
remove_punc_dict = dict((ord(punct), None ) for punct in string.punctuation)
def LemNormalize(text):
  return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))