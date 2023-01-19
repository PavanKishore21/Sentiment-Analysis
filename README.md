# Sentiment-Analysis

Sentiment analysis, also known as opinion mining, is a machine learning and NLP technique [61]. It can, as the name implies, examine the emotional tone conveyed by the author in any piece of text. Businesses use sentiment analysis tools to assess the sentiment value of their brands, goods, or services. Customer feedback analysis is one of the most widespread applications of sentiment analysis. Customers' emotions/sentiments can be analyzed and evaluated using sentiment analysis software. Sentiment analysis is used by data analysts in large companies to measure public opinion, track brand and product images, evaluate consumer experiences, and perform market research. There are many advantages of sentiment analysis such as recognizing your consumer base, developing and accessing a marketing campaign, improving customer service, crisis management, and increasing revenue from sales.

# Transformers

NLP’s Transformer is a new architecture that aims to solve tasks sequence-to-sequence while easily handling long-distance dependencies. Computing the input and output representations without using sequence-aligned RNNs or convolutions and it relies entirely on self-attention. Lets look in detail what are transformers.

![image](https://user-images.githubusercontent.com/78363747/213380972-f9072941-92e0-4d90-ad2b-ba2dd59b07d1.png)

# Why Transformers

Transformers are faster than RNN-based models as all the input is ingested once. Training LSTMs is harder when compared with transformer networks, since the number of parameters is a lot more in LSTM networks. Moreover, it’s impossible to do transfer learning in LSTM networks. Transformers are now state of the art network for seq2seq models.


# BERT

BERT is an open source machine learning framework for natural language processing (NLP). BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context. The BERT framework was pre-trained using text from Wikipedia and can be fine-tuned with question and answer datasets.

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. (In NLP, this process is called attention.)

Historically, language models could only read text input sequentially -- either left-to-right or right-to-left -- but couldn't do both at the same time. BERT is different because it is designed to read in both directions at once. This capability, enabled by the introduction of Transformers, is known as bidirectionality. 

# Labels Used

0 : Negative

1 : Positive

2 : Neutral
