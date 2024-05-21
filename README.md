# Comparative Analysis of TSLA Stock Price Prediction Models with Twitter Text Data

## Project Overview
This project aims to predict Tesla (TSLA) stock price movements using Twitter text data. We explore the effectiveness of advanced NLP models like LSTM and DistilBERT in analyzing sentiments and contents of tweets related to Tesla to forecast stock trends.

## Dataset
The dataset consists of tweets that have been labeled based on the corresponding stock price movement (up or down). Tweets were preprocessed to remove URLs, user mentions, and other non-essential text to ensure data cleanliness for modeling.

## Requirements
To run this project, you will need the following libraries:
- NumPy
- Pandas
- TensorFlow
- Scikit-Learn
- Transformers (Hugging Face)

You can install these dependencies via pip:
```bash
pip install numpy pandas tensorflow scikit-learn transformers
```

## Models Used

### LSTM Model
**Description:** LSTM networks were used to process sequences of embedded tweets, with dropout layers to mitigate overfitting.

**Code Snippet:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### DistilBERT Model
**Description:** Utilized the DistilBERT transformer model pre-trained on Twitter data for sentiment analysis before classifying stock movements.

**Code Snippet:**
```python
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import InputExample, InputFeatures

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize and encode sentences in the dataset
train_encodings = tokenizer(list_of_sentences, truncation=True, padding=True, max_length=128)
```

## How to Clone

```bash
git clone https://github.com/rafizbd912/tesla-stock-pred-v1.git
```




