import pickle
import re

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

STOP_WORDS = set([
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
    "won","wouldn",
    "na", "nee", "avan", "aval", "avanga", "inga", "anga",
    "enna", "epdi", "endha", "yaar", "oru", "alla", "illai",
    "da", "di", "pa", "ma", "la", "nu", "ku", "le"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

test_text = "you are so beautiful"
cleaned = clean_text(test_text)
proba = model.predict_proba([cleaned])[0]
classes = list(model.classes_)
harmful_idx = classes.index('harmful')
score = proba[harmful_idx]
verdict = "harmful" if score >= 0.5 else "non-harmful"

print(f"Text: '{test_text}'")
print(f"Cleaned: '{cleaned}'")
print(f"Harmful Score: {score:.4f}")
print(f"Verdict: {verdict}")
