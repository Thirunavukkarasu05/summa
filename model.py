# ==========================================
# MULTILINGUAL HARMFUL CONTENT DETECTOR
# Supports: English, Tamil, Tanglish, Hindi
# ==========================================

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score



# ==========================================
# TANGLISH / MULTILINGUAL KEYWORDS
# ==========================================

HARMFUL_TANGLISH = [
    # Tamil-English mix harmful phrases
    "mokka", "thevdiya", "punda", "otha", "naaye", "koothi",
    "velaiya po", "di poda", "da poda", "sunni", "paiyan",
    "oombu", "vaaya moodu", "paithiyam", "loosu",
    # Hindi harmful
    "bakwaas", "chutiya", "saala", "kamina", "harami",
    "gadha", "ullu", "bewakoof", "chup kar", "nikal",
    # Generic multilingual hate signals
    "maro", "jalao", "bhago", "maar", "khatam karo"
]

TONE_KEYWORDS = {
    "threatening": [
        "kill", "attack", "destroy", "burn", "shoot", "beat",
        "violence", "bomb", "stab", "hurt", "harm", "murder",
        "maro", "jalao", "maar", "adi", "thakku"
    ],
    "hateful": [
        "hate", "ban", "deport", "enemies", "invaders", "terrorist",
        "throw out", "not real", "inferior", "filth", "vermin",
        "naxal", "traitor", "anti-national"
    ],
    "doxxing": [
        "home address", "phone number", "personal information",
        "share details", "post his", "post her", "find out where",
        "school name", "college name", "doxx", "expose"
    ],
    "discriminatory": [
        "caste", "reservation", "minority", "religion", "community",
        "upper caste", "lower caste", "dalit", "muslim", "christian",
        "northeast", "migrant", "scheduled caste"
    ],
    "cyberbullying": [
        "hack", "private photos", "rumors", "reputation destroyed",
        "publicly humiliate", "disgraced", "silenced", "expelled",
        "blacklisted", "fake reviews"
    ],
    "neutral": []
}

INTENT_MAP = {
    "threatening": "Physical Harm / Violence",
    "hateful": "Hate Speech / Incitement",
    "doxxing": "Doxxing / Privacy Violation",
    "discriminatory": "Discrimination / Bias",
    "cyberbullying": "Cyberbullying / Harassment",
    "neutral": "No Harmful Intent"
}

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn"
])

# Add Tamil stopwords (romanized)
TAMIL_STOPWORDS = {
    "na", "nee", "avan", "aval", "avanga", "inga", "anga",
    "enna", "epdi", "endha", "yaar", "oru", "alla", "illai",
    "da", "di", "pa", "ma", "la", "nu", "ku", "le"
}
stop_words.update(TAMIL_STOPWORDS)

def detect_language(text):
    """Detect if text is Tanglish, Tamil, Hindi, or English."""
    text_lower = text.lower()
    tanglish_markers = ["poda", "ponga", "da", "di", "macha", "pa",
                        "kku", "nnu", "lla", "nga", "oru", "naan", "en"]
    hindi_markers = ["hai", "nahi", "karo", "aur", "mera", "tera",
                     "yaar", "bhai", "dost", "accha"]

    tanglish_score = sum(1 for m in tanglish_markers if f" {m} " in f" {text_lower} ")
    hindi_score = sum(1 for m in hindi_markers if f" {m} " in f" {text_lower} ")

    if tanglish_score >= 2:
        return "Tanglish"
    elif hindi_score >= 2:
        return "Hindi-English"
    elif any(w in text_lower for w in HARMFUL_TANGLISH):
        return "Regional"
    else:
        return "English"

def detect_tone(text):
    """Detect the dominant tone of the message."""
    text_lower = text.lower()
    scores = {}
    for tone, keywords in TONE_KEYWORDS.items():
        if tone == "neutral":
            continue
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[tone] = score

    max_tone = max(scores, key=scores.get)
    return max_tone if scores[max_tone] > 0 else "neutral"

def get_intent(tone):
    return INTENT_MAP.get(tone, "Unknown")

def clean_text(text):
    """Clean and normalize multilingual text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # Remove URLs
    text = re.sub(r"@\w+", "", text)                   # Remove mentions
    text = re.sub(r"#\w+", "", text)                   # Remove hashtags
    text = re.sub(r"[^a-z\s]", " ", text)             # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()           # Normalize spaces
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def extract_keywords(text):
    """Extract top harmful/significant keywords from text."""
    text_lower = text.lower()
    found = []
    all_keywords = []
    for keywords in TONE_KEYWORDS.values():
        all_keywords.extend(keywords)
    all_keywords.extend(HARMFUL_TANGLISH)

    for kw in set(all_keywords):
        if kw in text_lower:
            found.append(kw)
    return found[:5] if found else ["none"]

# ==========================================
# MAIN PIPELINE
# ==========================================

def build_model():
    df = pd.read_csv("dataset.csv")
    df['clean_text'] = df['text'].apply(clean_text)
    df['language'] = df['text'].apply(detect_language)
    df['tone'] = df['text'].apply(detect_tone)
    df['intent'] = df['tone'].apply(get_intent)
    df['keywords'] = df['text'].apply(extract_keywords)
    df['keywords_str'] = df['keywords'].apply(lambda x: ", ".join(x))

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=300, C=1.5, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Add toxicity scores
    proba = pipeline.predict_proba(df['clean_text'])
    harmful_idx = list(pipeline.classes_).index('harmful')
    df['toxicity_score'] = proba[:, harmful_idx]

    # Save outputs
    df.to_csv("processed_output.csv", index=False)
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Model saved to model.pkl")
    print("✅ processed_output.csv generated")
    return accuracy

if __name__ == "__main__":
    build_model()