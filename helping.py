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
import os




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

def least_busy_user(df):
    group_name = df['user'].iloc[0]
    df = df[(df['user'] != "group_notification") & (df['user'] != group_name)]
    least_active = df['user'].value_counts().idxmin()
    return f"The least active user is {least_active}."

def create_wordcloud(selected_user, df):
    stopwords_path = os.path.join("preprocessing", "stopwords.txt")
    with open(stopwords_path, 'r') as f:
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
    stopwords_path = os.path.join("preprocessing", "stopwords.txt")
    with open(stopwords_path, 'r') as f:
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
    model = joblib.load(os.path.join("vectorizer_files", "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join("vectorizer_files","tfidf_vectorizer.pkl"))
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

vectorizer = joblib.load(os.path.join("vectorizer_files", "tfidf_vectorizer_chatbot.pkl"))
model = joblib.load(os.path.join("vectorizer_files", "chatbot_model.pkl"))
intent_response_mapping = joblib.load(os.path.join("vectorizer_files","intent_response_mapping.pkl"))


# Intent-to-function mapping
intent_mapping = {
    0: emoji_helper,                 # (selected_user, df)
    1: most_busy_users,              # (df)
    2: least_busy_user,              # (df)
    3: fetch_stats,                  # (selected_user, df) --> For link count
    4: fetch_stats,                  # (selected_user, df)
    5: daily_timeline,               # (df)
    6: "With Whatseye, your chats are safe. Your privacy is our first priority.",  # Static response
    7: analyze_sentiment,            # (user_input)
    8: analyze_sentiment,            # (user_input)
    9: response_time_analysis,       # (df)
    10: response_time_analysis       # (df)
}

def get_chatbot_response(user_input, df, selected_user="Overall"):
    """
    Process user input, predict intent, and return the corresponding function's output.
    """
    # Ensure df is provided
    if df is None or df.empty:
        return "Error: Chat dataset is missing or empty."

    # Transform input using the vectorizer
    user_input_tfidf = vectorizer.transform([user_input])
    
    # Predict intent
    intent = model.predict(user_input_tfidf)[0]  

    if intent in intent_mapping:
        response_function = intent_mapping[intent]

        if callable(response_function):  
            if intent == 0:  
                emoji_df = response_function(selected_user, df)  # emoji_helper
                if emoji_df.empty:
                    result = "No emojis found in your chats."
                else:
                    top_emoji, count = emoji_df.iloc[0]  
                    result = f"Your most used emoji is {top_emoji} ({count} times)."

            elif intent == 1:  
                user_counts, _ = response_function(df)  # most_busy_users
                if user_counts.empty:
                    result = "No chat activity found."
                else:
                    most_active_user = user_counts.idxmax()
                    message_count = user_counts.max()
                    result = f"{most_active_user} is the most active user with {message_count} messages."

            elif intent == 2:  
                result = response_function(df)  # least_busy_user (already returns a string)

            elif intent == 3:  # Intent for total links
                num_messages, _, _, link_count = response_function(selected_user, df)  # fetch_stats
                result = f"A total of {link_count} links have been shared in your chats."

            elif intent == 4:  # Intent for total messages
                num_messages, _, _, _ = response_function(selected_user, df)  # fetch_stats
                result = f"The total number of messages in this chat is {num_messages}."

            elif intent == 5:  # Intent for most active chat time
                daily_data = response_function(selected_user, df)  # daily_timeline
            
                if daily_data.empty:
                    result = "No activity found in your chats."
                else:
                    # Find the day with the highest message count
                    busiest_day = daily_data.loc[daily_data['message'].idxmax(), 'only_date']
                    max_messages = daily_data['message'].max()
                    
                    result = f"Your chats were most active on {busiest_day} with {max_messages} messages."


            elif intent == 6:  
                result = response_function  # Static message: WhatsEye privacy assurance

            elif intent == 7:  # Intent for positivity percentage
                sentiment_df = integrate_sentiment_analysis(df)  # Apply sentiment analysis to entire chat
                
                if sentiment_df.empty:
                    result = "No messages found for sentiment analysis."
                else:
                    positive_count = (sentiment_df['sentiment'] == "Positive").sum()
                    total_messages = len(sentiment_df)
                    positivity_percentage = round((positive_count / total_messages) * 100, 2)

                    result = f"The positivity percentage of this group chat is {positivity_percentage}%."

            elif intent == 8:  # Intent for negativity percentage
                sentiment_df = integrate_sentiment_analysis(df)  # Apply sentiment analysis to entire chat

                if sentiment_df.empty:
                    result = "No messages found for sentiment analysis."
                else:
                    negative_count = (sentiment_df['sentiment'] == "Negative").sum()
                    total_messages = len(sentiment_df)
                    negativity_percentage = round((negative_count / total_messages) * 100, 2)

                    result = f"The negativity percentage of this group chat is {negativity_percentage}%."

            elif intent == 9:  # Intent for fastest responder
                response_time_df = response_function(df)  # response_time_analysis
                
                if response_time_df.empty:
                    result = "No response time data available."
                else:
                    fastest_responder = response_time_df.iloc[0, 0]  # First row (fastest)
                    result = f"{fastest_responder} is the fastest responder in your chats."

            elif intent == 10:  # Intent for slowest responder
                response_time_df = response_function(df)  # response_time_analysis
                
                if response_time_df.empty:
                    result = "No response time data available."
                else:
                    slowest_responder = response_time_df.iloc[-1, 0]  # Last row (slowest)
                    result = f"{slowest_responder} takes the longest to reply in your chats."


            elif intent == 11:  # Intent for most active chat day
                day_activity = week_activity_map(selected_user, df)
                
                if day_activity.empty:
                    result = "No activity data available."
                else:
                    most_active_day = day_activity.idxmax()  # Get the day with max messages
                    message_count = day_activity.max()  # Get the highest message count

                    result = f"Your most active day is {most_active_day} with {message_count} messages."


            else:
                result = "Error: Unhandled intent."  
        else:
            result = response_function  # Static message for intent 6

        return str(result)

    return "Sorry, I didn't understand that."


