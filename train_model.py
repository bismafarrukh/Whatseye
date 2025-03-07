import re
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Download stopwords (if not downloaded)
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)  # Keep punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load Dataset
def load_dataset():
    # Use a smaller sample for testing to avoid freezing
    file_path = "datasets/Dataset 11000 Reviews.tsv"
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8", on_bad_lines='skip', header=None)

    # Assign correct column names
    df.columns = ["sentiment", "message"]  # Assuming first column is sentiment, second is message

    # Convert sentiment labels to numeric values
    sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1, "neg": -1, "pos": 1, "neu": 0}
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)

    # Drop NaN values
    df = df.dropna(subset=["message", "sentiment"])

    # Ensure all messages are strings before applying `clean_text()`
    df["message"] = df["message"].astype(str)

    # Apply text cleaning
    df["cleaned_message"] = df["message"].apply(clean_text)

    # Use a smaller subset for testing purposes
    df = df.sample(n=1000, random_state=42)  # You can adjust n to use more data if necessary

    # Print dataset info
    print("\n🔹 Unique Sentiment Values After Mapping:", df["sentiment"].unique())
    print("\n🔹 Sample Data:\n", df.head())

    return df

# Train Model
def train_sentiment_model():
    df = load_dataset()

    # Print class distribution
    print("\n🔹 Sentiment Distribution:\n", df["sentiment"].value_counts())

    # Convert Text to TF-IDF Features with fewer features for testing
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords.words("english"))
    X = vectorizer.fit_transform(df["cleaned_message"])
    y = df["sentiment"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # Apply SMOTE to balance classes with a better resampling strategy
    smote = SMOTE(random_state=42, sampling_strategy=1.0)  # Make the minority class equal to the majority class
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


    # Hyperparameter tuning for SVM with a smaller grid search
    param_grid = {'C': [1], 'kernel': ['linear']}  # Reduced search space
    grid_search = GridSearchCV(SVC(), param_grid, cv=3)  # Reduced cross-validation folds for quicker execution
    grid_search.fit(X_resampled, y_resampled)

    # Get best model
    model = grid_search.best_estimator_

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n🔹 Model Accuracy:", accuracy)
    print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("\n✅ Model and vectorizer saved successfully!")

if __name__ == "__main__":
    train_sentiment_model()



