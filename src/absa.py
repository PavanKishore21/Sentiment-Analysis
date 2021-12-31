#!/usr/bin/env python
# coding: utf-8

# # Aspect Based Sentiment Analysis

# ## Importing data

# In[1]:


import tensorflow as tf
import sklearn
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import TFAutoModel
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import string
import re
import pandas as pd

df = pd.read_csv("data/Training/train - train.csv")


# In[2]:


df.shape


# In[3]:


df.head(10)


# ## Checking Null Values

# In[4]:


df.isna().sum()


# In[5]:


print((df['label'] == 0).sum())
print((df['label'] == 1).sum())
print((df['label'] == 2).sum())


# ## Preprocessing the data

# In[6]:


def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


string.punctuation


# ## Removal of punctuations and URLs

# In[7]:


df["text"] = df.text.map(remove_URL)  # map(lambda x: remove_URL(x))
df["text"] = df.text.map(remove_punct)


# In[8]:


df.head(10)


# ## Removing Stopwords

# In[9]:


nltk.download('stopwords')

# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
# has been programmed to ignore, both when indexing entries for searching and when retrieving them
# as the result of a search query.
stop = set(stopwords.words("english"))


def remove_stopwords(s):
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# In[10]:


df["text"] = df.text.map(remove_stopwords)


# In[11]:


df.head(10)


# In[12]:


df['text'] = df['text'] + " " + df['aspect']


# In[13]:


X = list(df['text'])
X


# In[14]:


y = list(df['label'])
y


# ## Splitting data for training and validation

# In[15]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=20)


# In[16]:


print('Training Data : ' + str(len(X_train)))
print('Validation Data : ' + str(len(X_val)))


# In[17]:


X_train


# ## Importing Transformers Pretrained Models

# In[18]:


# In[19]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[20]:


train_encodings = tokenizer(X_train,
                            truncation=True,
                            padding=True)
val_encodings = tokenizer(X_val,
                          truncation=True,
                          padding=True)


# In[21]:


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))


# In[22]:


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                              num_labels=3)


# ## Compiling the model

# In[23]:


optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
model.compile(optimizer=optimizer, loss=model.compute_loss,
              metrics=['accuracy'])


# ## Training the model

# In[25]:
print("================================ Training the model ==========================================")

model.fit(train_dataset.shuffle(100).batch(3),
          epochs=3,
          batch_size=32,
          validation_data=val_dataset.shuffle(100).batch(3))


# In[26]:


# model.save('absa_training')


# In[28]:


# model.save_pretrained("absa")


# In[ ]:
