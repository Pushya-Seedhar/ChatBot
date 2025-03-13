import pickle
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the trained model and vectorizer
with open('chatbot_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load intents
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Preprocessing function
def preprocess_text(text):
    return text.lower().strip()

# Chatbot function
def chatbot(input_text):
    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Test the chatbot
if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot(user_input)
        print(f"Chatbot: {response}")
