import random 
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from urlextract import URLExtract
import pandas as pd  # Required if you're using DataFrame operations



extract = URLExtract()
def fetch_stats(selected_user, df):
    words = []

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
#for messages
    num_messages = df.shape[0]
#for words
    words = []
    for message in df['message']:
        words.extend(message.split())


#for media
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

#for links
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))


    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    group_name = df['user'].iloc[0]
    df = df[(df['user'] != "group_notification") & (df['user'] != group_name)]
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0] * 100), 2).reset_index().rename(columns={'index': 'name', 'user':'percentage'})
    return x , df

def create_wordcloud(selected_user, df):
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.strip().str.contains(r'^\<Media omitted\>$', na=False)]

    def remove_stopwords(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return ' '.join(y)


    wc = WordCloud(width=500,height=500,background_color='white', max_words=200)
    temp['message']= temp['message'].apply(remove_stopwords)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user,df):
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read().split()


    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.strip().str.contains(r'^\<Media omitted\>$', na=False)]


    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df




def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    # Create a DataFrame and rename columns to 'emoji' and 'count'
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))), columns=['emoji', 'count'])

    return emoji_df



def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Load and preprocess WhatsApp chat data
def preprocess_chat(df):
    df['cleaned_message'] = df['message'].apply(clean_text)  # Clean messages
    return df


# Load trained sentiment model
def load_sentiment_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# Download required NLTK resources
nltk.download('vader_lexicon')

# Initialize sentiment analysis tools
sia = SentimentIntensityAnalyzer()
extract = URLExtract()


def analyze_sentiment(message):
    """Analyze sentiment of a given message using VADER."""
    sentiment_score = sia.polarity_scores(str(message))  # Ensure message is a string
    if sentiment_score['compound'] >= 0.05:
        return "Positive"
    elif sentiment_score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def integrate_sentiment_analysis(df):
    """Apply sentiment analysis using the trained ML model instead of VADER."""
    
    model, vectorizer = load_sentiment_model()  # Load the trained model
    df['cleaned_message'] = df['message'].astype(str).apply(clean_text)  # Preprocess messages
    features = vectorizer.transform(df['cleaned_message'])  # Convert to TF-IDF features
    predictions = model.predict(features)  # Predict sentiment
    
    sentiment_mapping = {1: "Positive", 0: "Neutral", -1: "Negative"}
    df['sentiment'] = [sentiment_mapping[pred] for pred in predictions]  # Map numerical labels to text labels
    
    return df

def find_most_ignored_person(df):
    """Identifies the user whose messages get ignored the most."""

    # Remove 'group_notification' and any group name
    group_name = df['user'].iloc[0]  # Assuming first row contains group name
    df = df[(df['user'] != "group_notification") & (df['user'] != group_name)]

    df = df.sort_values(by=['date'])
    df['time_diff'] = df.groupby('user')['date'].diff().dt.total_seconds()
    df['ignored'] = df['time_diff'] > 3600
    ignored_counts = df.groupby('user')['ignored'].sum().reset_index()
    ignored_counts = ignored_counts.rename(columns={'ignored': 'ignored_count'})

    return ignored_counts.sort_values(by='ignored_count', ascending=False)


def response_time_analysis(df):
    """Ranks users from fastest to slowest responders in a single table."""

    # Remove 'group_notification' and any group name
    group_name = df['user'].iloc[0]
    df = df[(df['user'] != "group_notification") & (df['user'] != group_name)]

    df = df.sort_values(by=['date'])
    df['response_time'] = df.groupby('user')['date'].diff().dt.total_seconds()
    response_times = df.groupby('user')['response_time'].mean().reset_index()
    response_times['Avg Response Time (hours)'] = response_times['response_time'] / 3600
    response_times = response_times.drop(columns=['response_time'])

    response_times_sorted = response_times.sort_values(by='Avg Response Time (hours)', ascending=True)
    total_users = len(response_times_sorted)

    response_times_sorted['Category'] = response_times_sorted.index.map(
        lambda x: "⚡ Fastest Responder" if x < total_users / 3 else
        "⚖️ Average Responder" if x < (2 * total_users) / 3 else
        "🐢 Slowest Responder"
    )

    return response_times_sorted


def detect_mood_swings(df):
    """Detects sudden emotional shifts in chat messages and prepares data for visualization."""
    group_name = df['user'].iloc[0]
    df = df[(df['user'] != "group_notification") & (df['user'] != group_name)]
    
    df = df[df['user'] != "group_notification"]
    df['date'] = pd.to_datetime(df['date'])

    sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_mapping)
    df['sentiment_shift'] = df.groupby('user')['sentiment_numeric'].diff().fillna(0)

    df['mood_swing'] = df['sentiment_shift'].apply(lambda x:
                                                   "🔴 Negative Shift" if x < -1 else
                                                   "🟢 Positive Shift" if x > 1 else
                                                   "⚪ Neutral")

    mood_swings_detected = df[df['mood_swing'] != "⚪ Neutral"]

    return mood_swings_detected


# Load trained model, vectorizer, and response mapping
vectorizer = joblib.load("tfidf_vectorizer_chatbot.pkl")
model = joblib.load("chatbot_model.pkl")
intent_response_mapping = joblib.load("intent_response_mapping.pkl")

def get_chatbot_response(user_input):
    user_input = [user_input]  # Convert to list for vectorization
    transformed_input = vectorizer.transform(user_input)
    predicted_intent = model.predict(transformed_input)[0]  # Get predicted intent
    
    # Get a response from mapping or return a default message
    if predicted_intent in intent_response_mapping:
        return random.choice(intent_response_mapping[predicted_intent])  # Choose a random response
    else:
        return "Sorry, I didn't understand that."
