
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import pairwise_distances
import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the mental illness chatbot data
with open('mental.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmer = nltk.stem.WordNetLemmatizer()

# Greeting inputs and responses
greet_inputs = ["hello", "hi", "greetings", "sup", "hey what's up"]
greet_responses = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

# Functions for chatbot logic
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)



def response(user_response):
    sentence_tokens = nltk.sent_tokenize(raw_doc)  # Define sentence_tokens here
    robo1_response = ''
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sentence_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robo1_response = "I am sorry! I don't understand you"
    else:
        robo1_response = sentence_tokens[idx]
    return robo1_response



# Web app routes
@app.route('/chat')
def index():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message'].lower()
    if user_message == 'bye':
        return jsonify({'response': 'Goodbye! Take care <3'})
    elif user_message in ['thanks', 'thank you']:
        return jsonify({'response': 'You are welcome..'})
    elif greet(user_message) is not None:
        return jsonify({'response': greet(user_message)})
    else:
        response_text = response(user_message)
        return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
