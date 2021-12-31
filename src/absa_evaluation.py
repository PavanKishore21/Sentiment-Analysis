
import tensorflow as tf
from nltk.corpus import stopwords
import nltk
import string
import re
from tensorflow import keras
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification


# In[3]:


import pandas as pd

test = pd.read_csv('data/Training/test - test.csv')


# In[4]:


test.shape


# In[5]:


test.head(10)


# In[6]:


test['text'] = test['text'] + ' ' + test['aspect']


# In[7]:


def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


string.punctuation


# In[8]:


test["text"] = test.text.map(remove_URL)  # map(lambda x: remove_URL(x))
test["text"] = test.text.map(remove_punct)


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


test["text"] = test.text.map(remove_stopwords)


# In[11]:


test_text = list(test['text'])
test_text[:10]


# In[12]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[13]:


test_encodings = tokenizer(test_text,
                           truncation=True,
                           padding=True)


# In[14]:


# print(test_encodings)


# In[16]:


# In[28]:


test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings)
))


# In[29]:

print("==========  Testing Data ============")

print(len(test_dataset))


# In[27]:


loaded_model = TFDistilBertForSequenceClassification.from_pretrained(
    'models/absa_training')


# In[30]:


test_output = loaded_model.predict(test_dataset)[0]
# print(test_output)


# In[49]:


test_prediction = []
for i in test_output:
    test_prediction.append(tf.nn.softmax(i, axis=0).numpy())


# In[47]:


def findclass(text):
    for i in range(0, 3):
        if text[i] == max(text):
            return i


# In[51]:


predictions = []

for text in test_prediction:
    predictions.append(findclass(text))


# In[59]:

print("==================  Predictions : ==================")
print(predictions)


# In[60]:


labels_df = pd.DataFrame(predictions, columns=['label'])


# In[67]:


test['label'] = predictions[:1000]


# In[71]:

print("=============  Predictions  =============== \n")
print(test.head(10))
