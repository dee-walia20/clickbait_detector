# Importing the libraries
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Import dataset and assign Input & Output
data = pd.read_csv('clickbait_data.csv')
X=data.headline
y=data.clickbait

#Custom Transformer built from sklearn for Text Processing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatizer=None, stopwords=None,token=True):
        self.token = token
        self.lemmatizer=lemmatizer
        self.stopwords=stopwords
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.token:
            nltk.download('averaged_perceptron_tagger')
            X=X.apply(lambda x: re.sub('\d+',"",x))
            X=X.apply(lambda x: re.sub('\W'," ",x))
            X=X.apply(lambda x: re.sub("  "," ",x))
            X=X.apply(lambda x: x.lower())
            self.lemmatizer=WordNetLemmatizer()
            nltk.download("stopwords")
            self.stopwords=stopwords.words('english')
            X=X.apply(lambda x: " ".join([self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word)) 
                          for word in x.split() if word not in self.stopwords]))
        return X
    def _get_wordnet_pos(self, word):
        tag=nltk.pos_tag([word])[0][1][0].upper()
        tag_dict={'J':wordnet.ADJ,
                  'N':wordnet.NOUN,
                  'V':wordnet.VERB,
                  'R':wordnet.ADV
                 }
        return tag_dict.get(tag,wordnet.NOUN)

#Splitting Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=10)

#Creating the pipeline for End to end ML model

tp=TextPreprocessor()
tf=TfidfVectorizer()
model=MultinomialNB()
pipeline=make_pipeline(tp,tf,model)

#Fitting model with trainig data
pipeline.fit(X_train, y_train)

# Saving model to disk
pickle.dump(pipeline, open('pipeline.pkl','wb'), protocol=-1)

# Loading model to check performance on Test set

if __name__ == "__main__":
    saved_pipeline = pickle.load(open('pipeline.pkl','rb'))
    print("Classitication Report\n",classification_report(y_test, saved_pipeline.predict(X_test)))