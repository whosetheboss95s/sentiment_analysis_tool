import torch
from transformers import pipeline
from flask import Flask, render_template, request

app = Flask(__name__)

def analyze_sentiment_nltk(message):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(message)

    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment_transformers(message):
    classifier = pipeline('sentiment-analysis')
    result = classifier(message)[0]
    return result['label']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_message = request.form['message']
        nltk_sentiment = analyze_sentiment_nltk(user_message)
        transformers_sentiment = analyze_sentiment_transformers(user_message)

        return render_template('index.html', message=user_message, nltk_sentiment=nltk_sentiment, transformers_sentiment=transformers_sentiment)

    return render_template('index.html', message=None, nltk_sentiment=None, transformers_sentiment=None)

if __name__ == "__main__":
    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure transformers library is installed
    try:
        from transformers import pipeline
    except ImportError:
        print("Please install the transformers library: pip install transformers")

    # Download nltk resources
    nltk.download('vader_lexicon')

    app.run(debug=True)
