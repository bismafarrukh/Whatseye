import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

nltk.download("stopwords")

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Define dataset path
dataset_path = os.path.join("datasets", "chatbot_dataset.csv" )

# Load the dataset
df = pd.read_csv(dataset_path)

# Ensure the dataset has "Message" and "Intent"
required_columns = {"Message", "Intent"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain {required_columns} columns. Found: {df.columns}")

# Drop missing values
df = df.dropna()

# Apply text cleaning
df["Message"] = df["Message"].apply(clean_text)

# Debug: Print dataset info
print("\n🔹 Dataset Sample:\n", df.head())

# Ensure dataset is not empty after cleaning
if df.empty:
    raise ValueError("Dataset is empty after preprocessing. Check data quality.")

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
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

print("\n✅ Chatbot model and vectorizer saved successfully!")

