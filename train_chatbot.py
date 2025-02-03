import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import json
from nltk.corpus import stopwords

# Load dataset (update path if necessary)
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

nltk.download("stopwords")

# Apply cleaning
df["cleaned_text"] = df["User_Query"].apply(clean_text)
df["response_text"] = df["Chatbot_Response"].apply(clean_text)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["cleaned_text"])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
output_sequences = tokenizer.texts_to_sequences(df["response_text"])

# Pad sequences
input_padded = pad_sequences(input_sequences, maxlen=20, padding="post")
output_padded = pad_sequences(output_sequences, maxlen=20, padding="post")

# Save tokenizer
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer.word_index, f)
