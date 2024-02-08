from flask import Flask, render_template, request
import json
import random
import torch
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

app = Flask(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = './model/data.pth'
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()





def chat(inp):
    while True:
        sentence = inp
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > .75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    return random.choice(intent["responses"])
        else:
            return 'Sorry. I do not understand'
   

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chat(userText))


if __name__ == "__main__":
    app.run()
