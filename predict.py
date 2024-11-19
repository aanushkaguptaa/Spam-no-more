from flask import Flask, request, jsonify, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import abort
from werkzeug.exceptions import HTTPException

# T5 Model for summarization
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(model_name)  # Renamed to summarizer_model

def get_dynamic_length(text_length, min_ratio=0.3, max_ratio=0.5, min_length=10, max_length=150):
    """
    Calculate dynamic summary length based on input text length
    
    Parameters:
    - text_length: Length of input text
    - min_ratio: Minimum ratio of summary to original text (default 0.3 or 30%)
    - max_ratio: Maximum ratio of summary to original text (default 0.5 or 50%)
    - min_length: Absolute minimum summary length
    - max_length: Absolute maximum summary length
    """
    # Calculate target length as 30-50% of original
    target_length = int(text_length * min_ratio)
    
    # Ensure length is within absolute bounds
    target_length = max(min_length, min(target_length, max_length))
    
    # Calculate minimum length as 60% of target length
    min_target_length = max(min_length, int(target_length * 0.6))
    
    return min_target_length, target_length

@lru_cache(maxsize=1000)

def summarize(text, min_length=10, max_length=150):
    """Generate dynamic-length summary using T5"""
    # Calculate target lengths
    text_length = len(text.split())
    min_target_length, target_length = get_dynamic_length(
        text_length,
        min_length=min_length,
        max_length=max_length
    )
    
    # Prefix text with "summarize: " for T5 model
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary with adjusted parameters
    summary_ids = summarizer_model.generate(
        inputs,
        max_length=target_length,
        min_length=min_target_length,
        length_penalty=2.0,          # Increased to encourage longer summaries
        num_beams=4,
        temperature=0.7,             # Added for more diverse outputs
        do_sample=True,              # Enable sampling
        early_stopping=True,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

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

@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({
        "error": str(e),
        "message": e.description
    }), e.code

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    try:
        if not request.is_json:
            abort(400, description="Request must be JSON")
            
        input_sms = request.json.get('text')
        if not input_sms:
            abort(400, description="Missing 'text' field")
            
        if len(input_sms) > 5000:  # Increased limit since we handle length dynamically
            abort(400, description="Text too long (max 5000 chars)")
        
        # Get optional summary length parameters
        min_length = request.json.get('min_length', 10)
        max_length = request.json.get('max_length', 150)
        
        # Validate length parameters
        if not (10 <= min_length <= max_length <= 500):
            abort(400, description="Invalid length parameters. min_length must be ≥10, max_length must be ≤500")
            
        summary = summarize(input_sms, min_length, max_length)
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = spam_model.predict(vector_input)[0]
        
        return jsonify({
            'result': 'Spam' if result == 1 else 'Ham',
            'summary': summary,
            'confidence': float(spam_model.predict_proba(vector_input)[0][result]),
            'original_length': len(input_sms.split()),
            'summary_length': len(summary.split())
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)