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
  //greet function
  greet_inputs=('hello','hi','whatsap','how are you?')
greet_responses=('hi','hey','hey there','there there')
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
  robo1_response=''
  TfidfVect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
  tfidf = TfidfVec.fit_transform(sentence_tokens)
  vals = cosine_similarity(tfidf[-1],tfidf)
  idx = vals.argsort()[0][-2]
  flat = vals.flatten
  flat.sort()
  req_tfidf = flat[-2]
  if (req_tfidf==0):
    robo1_response =  robo1_response +"Sorry , unable to understand you"
    return robo1_response
  else:
     robo1_response =  robo1_response + sentence_tokens(idx)
     return  robo1_response

flag=True
print('hello! I am a learning chatbot start typing your text after greeting me and end the convo by saying bye')
while(flag==True):
  user_response=input()
  user_response=user_response.lower()
  if(user_response != 'bye'):
    if(user_response =='Thankyou' or user_response =='Thanks'):
      flag = False
      print('bot you are welcome!')
    else:
      if(greet(user_response) != None):
        print('bot'+ greet(user_response))
      else:
        sentence_tokens.append(user_response)
        word_tokens = word_tokens + nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        print('bot:', end='')
        print(response(user_response))
        sentence_tokens.remove(user_response)
  else:
       flag = False
       print('bot: Goodbye')     







