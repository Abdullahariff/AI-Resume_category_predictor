import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download once if not already
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]

    return " ".join(lemmatized)
