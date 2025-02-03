import joblib
from transformers import pipeline

# Load the trained model
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load an AI text generation model (like GPT-2 or T5)
text_generator = pipeline("text-generation", model="distilgpt2")


def chatbot():
    print("Chatbot: Hello! Ask me anything. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip().lower()
        if user_input == "exit":
            print("Chatbot: Goodbye!")
            break

        # Convert user input to numerical features
        user_vector = vectorizer.transform([user_input])

        # Predict intent
        intent_idx = model.predict(user_vector)[0]
        intent = label_encoder.inverse_transform([intent_idx])[0]

        # Generate a response dynamically
        response = text_generator(f"User asked: {user_input} Intent: {intent}", max_length=50, do_sample=True)[0]["generated_text"]

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
