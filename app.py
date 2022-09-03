import streamlit as st
import numpy as np
import nltk
# download some library which hold in nltk only first time
nltk.download('punkt') #punctuation
nltk.download('wordnet')#for lemmatization
nltk.download('stopwords')#for stopwords corpus
nltk.download('omw-1.4')

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re,string
from keras.preprocessing import text, sequence
from string import punctuation
import pickle
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

# load pickle file
with open("tokenizer.pickle", 'rb') as handle:
    tok = pickle.load(handle)
#load model
model = load_model('model.h5')
#model = pickle.load(open('model.h5','rb'))


#Test cleaning
#remove URLS 
def remove_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r" ", text)

#remove HTMLS tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r" ", text)

#remove numbers
def remove_numbers(text):
    removed_numbers = text.replace(r'\d+','')
    return removed_numbers

#remove emails 
def remove_emails(text):
    no_emails = text.replace(r"\S*@\S*\s?",'')
    return no_emails

#remove emoi
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

lst=['ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", "won't", 'wouldn', "wouldn't",
'are not','could not','would not','did not','does not','did not','was not','wasnt','does not','had not','have not','is not','might not','must not','need not','shall not','was not','no so','had not','wont',"won't",'do not','is not','isnt',"isn't",'not','never','am not']
#apply decontraction
def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)
    text = re.sub(r"did\'t", " did not", text)
    text = re.sub(r"would\'t", " would not", text)
    text = re.sub(r"ain\'t", " are not", text)
    text = re.sub(r"does\'t", " does not", text)
    text = re.sub(r"havn\'t", " have not", text)
    text = re.sub(r"is\'t", " is not", text)

    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"wasn\'t", " was not", text)
    text = re.sub(r" u ", " you ", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"   "," ",text)
    text = re.sub(r"  "," ",text)
    return text 

def clean_text(text):
    #1. convert text into lower case
    text=text.lower()
    
    #2. Replace All negative words by not  
    lst1=set(lst)
    for i in lst1:
        if i in text:
            text=text.replace(i,'not')
            text=re.sub(r'not\'t', 'not',text)
            text = re.sub(r"  "," ",text)
    
    #2.word_tokenize
    text=text.strip()
    text=word_tokenize(text)
    
    #3.remove negativity from stopwords
    sw=stopwords.words('english')
    unwanted_ele=set(lst)
    updated_sw = [ele for ele in sw if ele not in unwanted_ele]
    
    #3.remove punctuation and stopwords
    words=updated_sw+list(string.punctuation)+list(["would","could","should","will","have","had"])
    text=[word for word in text if word not in words]
    
    #4. use only spcl characters consider only alphbets
    text=[word for word in text if word.isalpha() ==True]
    
    #5. apply lamatization to covert all plurals into singular
    lemma=WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    #text=set(text)  #remove duplicates words
    
    #join text
    text=' '.join(text)

    return text



st.title("Sentiment Analysis Movie Reviews")

text = st.text_area("Enter the Review")
if st.button("clear text"):
    text=text.replace(text,"")


if st.button('Predict'):

    # 1. preprocess
    text=remove_urls(text)
    text=remove_emails(text)
    text=remove_numbers(text)
    text=remove_html(text)
    text= decontraction(text)
    text=clean_text(text)
    text
    # 2. tokenize
    tok_text = tok.texts_to_sequences([text])
    # 3. padding
    padd_tok_text1 = pad_sequences(tok_text,  maxlen=295)
    #predict
    result=model.predict(padd_tok_text1)[0]
    #display
    st.write(result)
    #predict categ
    if result>0.5:
        st.write("Positive 😊")
    else:
        st.write("Negative 😔")
