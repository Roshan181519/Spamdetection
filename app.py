from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Route to serve the UI
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_content = request.form.get('message')  # Get message from the form
        processed_email = preprocess_text(email_content)
        tfidf_features = vectorizer.transform([processed_email])
        prediction = model.predict(tfidf_features)
        result = 'spam' if prediction[0] == 1 else 'ham'
        return render_template('index.html', prediction=result, email=email_content)
    except Exception as e:
        return render_template('index.html', error=str(e))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
