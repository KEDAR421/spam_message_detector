import pandas as pd
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Text Preprocessing Function ---
# This is the same function we will use in the Flask app
ps = PorterStemmer()

def preprocess_text(text):
    """
    Cleans and preprocesses raw text for the model.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()                   # Convert to lowercase
    text = text.split()                   # Tokenize
    
    # Remove stopwords and apply stemming
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    
    return ' '.join(text)

# --- Main Training Logic ---
print("1. Loading dataset...")
# Load dataset (make sure spam.csv exists)
data = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print("2. Preprocessing all messages...")
# Apply the preprocessing function to the entire message column
data['processed_message'] = data['message'].apply(preprocess_text)

print("3. Splitting data...")
# Train-test split on the PROCESSED messages
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_message'], data['label'], test_size=0.2, random_state=42
)

print("4. Vectorizing text...")
# Vectorizer will now learn from the stemmed/cleaned text
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

print("5. Training Naive Bayes model...")
# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("6. Saving model and vectorizer...")
# Save trained model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Training complete! New model.pkl & vectorizer.pkl have been saved.")

