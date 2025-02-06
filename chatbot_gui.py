import tkinter as tk
from tkinter import *
import json
import numpy as np
import random
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load Model and Data
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

# Preprocess User Input
def clean_up_sentence(sentence):
    sentence_words = sentence.lower().split()  # Basic split
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict Intent
def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Generate Chatbot Response
def chatbot_response(msg):
    intents_list = predict_class(msg)
    intent = intents_list[0]["intent"] if intents_list else "fallback"

    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

    return "I don't understand that."

# GUI Function
def send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# Create GUI Window
base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=False, height=False)

# Chat window
ChatLog = Text(base, bd=0, bg="black", fg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

# Scrollbar
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Entry box
EntryBox = Text(base, bd=0, bg="black", fg="white", width="29", height="5", font="Arial")

# Bind `Enter` key to send messages
def enter_pressed(event):
    send()

EntryBox.bind("<Return>", enter_pressed)

# Send button
SendButton = Button(base, text="Send", font=("Verdana", 12, 'bold'), width="12", height="5", bg="#32de97", command=send)

# Place components
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=276, y=401, height=90)

base.mainloop()
