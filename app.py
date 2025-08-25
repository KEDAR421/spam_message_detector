from flask import Flask, render_template, request
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------- NLTK SETUP (Render-friendly) ----------------
# Store NLTK data inside the repo path so it survives across runs.
NLTK_DIR = "/opt/render/project/src/.nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Download required datasets (idempotent)
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{'punkt_tab' if pkg=='punkt_tab' else 'punkt'}") if "punkt" in pkg \
            else nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)
# --------------------------------------------------------------

# Initialize Flask application
app = Flask(__name__)

# --- Load Model & Vectorizer ---
try:
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    clf = None
    tfidf = None

# --- Text Preprocessing Function ---
ps = PorterStemmer()
_stopwords = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    alnum = [t for t in tokens if t.isalnum()]
    stemmed = [ps.stem(t) for t in alnum if t not in _stopwords and t not in string.punctuation]
    return " ".join(stemmed)

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = "An error occurred."
    if clf and tfidf:
        try:
            message = request.form.get('message', '')
            if message.strip():
                transformed_sms = preprocess_text(message)
                vector_input = tfidf.transform([transformed_sms])
                result = clf.predict(vector_input)[0]
                prediction_text = "🚨 SPAM!" if result == 1 else "✅ Not Spam."
            else:
                prediction_text = "Please enter a message to analyze."
        except Exception as e:
            prediction_text = f"Error during prediction: {e}"
    else:
        prediction_text = "Model not loaded. Cannot make a prediction."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
