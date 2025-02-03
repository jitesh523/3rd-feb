import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import pickle

# ✅ Step 1: Load Dataset
file_path = "/Users/neha/Documents/chatbot/dialogs.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ✅ Split dataset into questions and responses
conversations = [line.strip().split("\t") for line in lines if "\t" in line]
df = pd.DataFrame(conversations, columns=["question", "response"])

# ✅ Step 2: Preprocess Text
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    text = text.lower().strip()
    return " ".join([word for word in text.split() if word not in stop_words])

df["clean_question"] = df["question"].apply(clean_text)
df["clean_response"] = df["response"].apply(clean_text)

# ✅ Step 3: Convert Text to Vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["clean_question"]).toarray()
y = df["clean_response"].values

# ✅ Step 4: Encode Responses
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ✅ Save Tokenizer & Label Encoder
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# ✅ Step 5: Convert to PyTorch Dataset
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = ChatDataset(X, y_encoded)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ✅ Step 6: Define RNN Model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x.unsqueeze(1))
        output = self.fc(hidden.squeeze(0))
        return output

input_size = X.shape[1]
hidden_size = 64
output_size = len(label_encoder.classes_)

model = ChatbotModel(input_size, hidden_size, output_size)

# ✅ Step 7: Train Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✅ Training Chatbot Model...")

for epoch in range(20):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

# ✅ Save Trained Model
torch.save(model.state_dict(), "chatbot_model.pth")
pickle.dump(df["clean_question"].tolist(), open("questions.pkl", "wb"))
pickle.dump(df["clean_response"].tolist(), open("responses.pkl", "wb"))

print("✅ Model trained and saved successfully!")
