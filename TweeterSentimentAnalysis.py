
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.model_selection import GridSearchCV
import preprocessor as p
import csv, string


# In[3]:

df_train = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", names =["Class", "ID", "Date", "ToBeDeleted","Name", "UnprocessedText"])
df_test = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', encoding = "ISO-8859-1", names =["Class","Some", "ID", "Date", "ToBeDeleted", "UnprocessedText"])
df_train.drop(['ID', 'Date', 'ToBeDeleted', 'Name'], axis=1, inplace=True)
df_test.drop(['Some','ID', 'Date', 'ToBeDeleted'], axis=1, inplace=True)


# In[4]:

def clean(df,cleaned_file_path):
    with open(cleaned_file_path, 'w') as cleanedDatacsvfile:
        cleanedDatawriter = csv.writer(cleanedDatacsvfile)
        for r in df['UnprocessedText']:
            #polarity = r[0].encode('utf-8', errors='replace')
            #running into problem in this line with encoding, tried utf-8 and latin-1
            cleanedText = p.clean(r).encode('utf-8', errors='replace')
            row = [cleanedText]
            cleanedDatawriter.writerow(row)

def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

clean(df_train, 'trainingandtestdata/train_preprocessed.csv')
clean(df_test, 'trainingandtestdata/test_preprocessed.csv')


# In[7]:

#df_train_preprocessed = pd.read_csv('trainingandtestdata/train_preprocessed.csv', encoding = "ISO-8859-1", names=['text'])
#df_train_preprocessed

#df_test_preprocessed = pd.read_csv('trainingandtestdata/test_preprocessed.csv', encoding="ISO-8859-1", names=['text'])
#df_test_preprocessed


# In[14]:

# del df_train_preprocessed['random']
#df_train_preprocessed['text'] = df_train_preprocessed['text'].map(lambda x: x.lstrip('b'))
#df_train_preprocessed['text'] = df_train_preprocessed['text'].apply(remove_punctuation)
# #df_test
# #df_test_preprocessed
#df_train = pd.concat([df_train, df_train_preprocessed], axis=1)

df_test_preprocessed = pd.read_csv('trainingandtestdata/test_preprocessed.csv', encoding = "ISO-8859-1", names=['text'])
# del df_test_preprocessed['random']
df_test_preprocessed['text'] = df_test_preprocessed['text'].map(lambda x: x.lstrip('b'))
df_test_preprocessed['text'] = df_test_preprocessed['text'].apply(remove_punctuation)
# #df_test
# #df_test_preprocessed
df_test = pd.concat([df_test, df_test_preprocessed], axis=1)

# del df_test['Some']
# del df_test['ID']
# del df_test['Date']
# del df_test['ToBeDeleted']


# In[22]:

df_train[df_train["Class"]==4]


# In[17]:

import pandas as pd
#df_train.to_csv("finalprocessing.csv")
#df_test.to_csv("finalprocessing_test.csv")

df_train = pd.read_csv('finalprocessing.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('finalprocessing_test.csv', encoding='ISO-8859-1')

df_train = df_train[df_train.text.notnull()]
df_train


# In[19]:

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(stop_words='english', lowercase=True)
X_train_counts = count_vect.fit_transform(df_train['text'])
X_train_counts.shape


# In[34]:

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[49]:

from sklearn.naive_bayes import MultinomialNB
from numpy import array
import numpy as np

df_list = array(df_train['Class'].tolist())
df_list

#x= np.random.randint(5, size=(6,100))
#y = np.array([1,2,3,4,5,6])

# dflist = df_train['Class'].tolist()
# df_final = df_train.as_matrix(columns=['Class'])
# X_train_tfidf
clf = MultinomialNB()
clf.fit(X_train_tfidf,df_list)



#print (clf.predict(x[2:3]))
#dflist


# In[54]:

test_text = df_test['text'][0]
testing = [test_text]
X_new_counts = count_vect.transform(testing)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


# In[55]:

predicted


# In[58]:

from sklearn.pipeline import Pipeline

text_clf = Pipeline ([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
text_clf = text_clf.fit(df_train['text'], df_list)


# In[63]:

predicted = text_clf.predict(df_test['text'])


# In[64]:

df_test_list = array(df_test['Class'].tolist())
np.mean(predicted ==  df_test_list)


# In[61]:

df_test = df_test[df_test.Class != 2]


# In[62]:

df_test


# In[69]:

from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-2, n_iter=10, random_state=42)),
                    ])

text_clf = text_clf.fit(df_train['text'], df_list)



# In[70]:

predicted = text_clf.predict(df_test['text'])
df_test_list = array(df_test['Class'].tolist())
np.mean(predicted ==  df_test_list)


# In[ ]:



