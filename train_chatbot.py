import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load Dataset
df = pd.read_csv("chatbot_dataset.csv").dropna()

# Ensure the dataset has "Message", "Intent", and "Response"
if "Response" not in df.columns:
    raise ValueError("Dataset must contain 'Message', 'Intent', and 'Response' columns.")

# Apply text cleaning
df["Message"] = df["Message"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["Message"])
y = df["Intent"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n🔹 Model Accuracy:", accuracy)
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer_chatbot.pkl")
joblib.dump(model, "chatbot_model.pkl")

# Save intent-response mapping
intent_response_mapping = df.groupby("Intent")["Response"].apply(list).to_dict()
joblib.dump(intent_response_mapping, "intent_response_mapping.pkl")

print("\n✅ Chatbot model, vectorizer, and response mapping saved successfully!")




