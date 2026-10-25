"""
Text Preprocessing Pipeline for Customer Review Sentiment Analysis.
Handles HTML removal, contraction expansion, negation-preserving stop-word removal,
and WordNet lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure required NLTK resources are downloaded safely without crashing
def _download_nltk_resources():
    required_packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet.zip", "wordnet"),
        ("corpora/omw-1.4.zip", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab")
    ]
    for resource_path, pkg_name in required_packages:
        try:
            nltk.data.find(resource_path)
        except (LookupError, IndexError):
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception as e:
                print(f"[Warning] Could not auto-download NLTK package '{pkg_name}': {e}")

_download_nltk_resources()

# Initialize Lemmatizer
try:
    _lemmatizer = WordNetLemmatizer()
except Exception:
    _lemmatizer = None

# Build Negation-Preserving Stopwords Set
# Standard stopwords list often removes critical negative indicators like "not", "no", "never",
# which drastically flips sentiment polarity in sentiment analysis tasks.
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "nobody",
    "nowhere", "nothing", "barely", "hardly", "scarcely", "cannot"
}

try:
    _raw_stopwords = set(stopwords.words("english"))
    # Preserve key negation polarity words
    CUSTOM_STOPWORDS = _raw_stopwords - NEGATION_WORDS
except Exception:
    CUSTOM_STOPWORDS = {
        "the", "a", "an", "in", "on", "at", "to", "for", "with", "is", "was",
        "are", "were", "and", "or", "of", "it", "this", "that"
    }

# Common English Contractions Mapping for Sentiment Analysis
CONTRACTIONS_MAP = {
    r"can't": "cannot",
    r"won't": "will not",
    r"n't": " not",
    r"i'm": "i am",
    r"it's": "it is",
    r"he's": "he is",
    r"she's": "she is",
    r"that's": "that is",
    r"what's": "what is",
    r"where's": "where is",
    r"there's": "there is",
    r"they're": "they are",
    r"we're": "we are",
    r"you're": "you are",
    r"i've": "i have",
    r"we've": "we have",
    r"you've": "you have",
    r"they've": "they have",
    r"i'll": "i will",
    r"you'll": "you will",
    r"he'll": "he will",
    r"she'll": "she will",
    r"we'll": "we will",
    r"they'll": "they will",
    r"i'd": "i would",
    r"you'd": "you would",
    r"he'd": "he would",
    r"she'd": "she would",
    r"we'd": "we would",
    r"they'd": "they would"
}

def expand_contractions(text: str) -> str:
    """Expand conversational contractions to preserve negations and word forms."""
    lowered = text.lower()
    for pattern, replacement in CONTRACTIONS_MAP.items():
        lowered = re.sub(pattern, replacement, lowered)
    return lowered

def clean_text(text: str) -> str:
    """
    Cleans raw customer review text through a multi-stage NLP cleaning pipeline:
    1. Type validation and basic conversion
    2. HTML tag removal (<br />, <p>, etc.)
    3. URL and Email stripping
    4. Contraction expansion (e.g. didn't -> did not)
    5. Removal of non-alphabetic characters
    6. Tokenization and Negation-Preserving stop-word filtering
    7. Lemmatization via WordNet
    """
    if text is None:
        return ""
    
    # Ensure string
    text = str(text)
    
    # 1. Expand contractions before punctuation stripping
    text = expand_contractions(text)
    
    # 2. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Strip URLs and Emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    
    # 4. Remove special characters and digits, keep spaces and letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 5. Tokenize by whitespace
    tokens = text.lower().split()
    
    # 6. Filter stopwords (preserving negations) & Lemmatize
    cleaned_tokens = []
    for token in tokens:
        if len(token) > 1 and token not in CUSTOM_STOPWORDS:
            if _lemmatizer:
                try:
                    token = _lemmatizer.lemmatize(token)
                except Exception:
                    pass
            cleaned_tokens.append(token)
            
    return " ".join(cleaned_tokens)

def clean_corpus(texts, show_progress: bool = False):
    """Clean an iterable of texts with optional progress reporting."""
    cleaned = []
    total = len(texts) if hasattr(texts, "__len__") else None
    
    for i, t in enumerate(texts):
        cleaned.append(clean_text(t))
        if show_progress and total and (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{total} texts...")
            
    return cleaned

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer to seamlessly integrate
    text cleaning into an sklearn Pipeline.
    """
    def __init__(self, show_progress: bool = False):
        self.show_progress = show_progress

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "tolist"):
            items = X.tolist()
        elif isinstance(X, (list, tuple)):
            items = X
        else:
            items = list(X)
        return clean_corpus(items, show_progress=self.show_progress)
