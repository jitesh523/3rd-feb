import torch
import pickle
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# ✅ Step 1: Load Pretrained Model and Tokenizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
questions = pickle.load(open("questions.pkl", "rb"))
responses = pickle.load(open("responses.pkl", "rb"))

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x.unsqueeze(1))
        output = self.fc(hidden.squeeze(0))
        return output

input_size = len(vectorizer.get_feature_names_out())
hidden_size = 64
output_size = len(label_encoder.classes_)

model = ChatbotModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()

# ✅ Step 2: Interactive Chatbot with Memory
memory = []  # Stores last 5 messages for context

def get_best_response(user_input):
    """Finds the best response using the trained model."""
    input_vector = vectorizer.transform([user_input]).toarray()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_index = torch.argmax(output).item()
    return responses[predicted_index]

def chat():
    print("🤖 Chatbot: Hello! I can remember past conversations. (Type 'exit' to stop)")

    while True:
        user_input = input("You: ").strip().lower()
        if user_input == "exit":
            print("🤖 Chatbot: Goodbye!")
            break

        # ✅ Remember past conversations
        memory.append(user_input)
        if len(memory) > 5:
            memory.pop(0)

        # ✅ Get AI-generated response
        best_response = get_best_response(" ".join(memory))

        print(f"🤖 Chatbot: {best_response}")

# ✅ Step 3: Run Chatbot
chat()
