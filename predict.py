from flask import Flask, request, jsonify, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# T5 Model for summarization
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(model_name)  # Renamed to summarizer_model

def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    # Modified parameters for more concise summaries
    summary_ids = summarizer_model.generate(
        inputs,
        max_length=50,  # Reduced from 150
        min_length=10,  # Reduced from 40
        length_penalty=1.0,  # Reduced from 2.0
        num_beams=4,
        early_stopping=True,
        repetition_penalty=2.5,  # Added to prevent repetition
        no_repeat_ngram_size=2  # Added to prevent repetition
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

app = Flask(__name__)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Models for spam detection
tfidf = pickle.load(open('vectorizer.pkl','rb'))
spam_model = pickle.load(open('model.pkl','rb'))  # Renamed to spam_model

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.json['text']
    summary = summarize(input_sms)
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = spam_model.predict(vector_input)[0]  # Use spam_model here
    if result == 1:
        return jsonify({'result': 'Spam', 'summary': summary})
    else:
        return jsonify({'result': 'Not Spam', 'summary': summary})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)