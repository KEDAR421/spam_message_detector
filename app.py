from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Initialize Flask application
app = Flask(__name__)

# --- Load Model & Vectorizer ---
try:
    clf = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    clf = None
    tfidf = None

# --- Text Preprocessing Function (Your new logic) ---
ps = PorterStemmer()

def preprocess_text(text):
    """
    Cleans and preprocesses raw text using the new, detailed logic.
    """
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Collect alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation, then apply stemming
    text = [
        ps.stem(i) for i in y 
        if i not in stopwords.words('english') and i not in string.punctuation
    ]
    
    return " ".join(text)

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the HTML form."""
    prediction_text = "An error occurred."
    if clf and tfidf:
        try:
            message = request.form['message']
            if message.strip():
                transformed_sms = preprocess_text(message)
                vector_input = tfidf.transform([transformed_sms])
                result = clf.predict(vector_input)[0]
                
                if result == 1:
                    prediction_text = "🚨 SPAM!"
                else:
                    prediction_text = "✅ Not Spam."
            else:
                prediction_text = "Please enter a message to analyze."
        except Exception as e:
            prediction_text = f"Error during prediction: {e}"
    else:
        prediction_text = "Model not loaded. Cannot make a prediction."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
