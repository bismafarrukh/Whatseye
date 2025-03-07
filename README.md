# WhatsEye 👁‍🗨 - WhatsApp Chat & Sentiment Analysis Tool
📊 Analyze WhatsApp conversations for sentiment, engagement, and communication patterns in both English and Roman Urdu.

# 🔍 Overview
WhatsEye is a powerful NLP-based analytics tool designed to extract insights from WhatsApp chat exports. It performs sentiment analysis, tracks message engagement, detects mood swings, and provides detailed visualizations to understand user behavior in chat conversations.

Whether you’re analyzing group chats, personal conversations, or business communications, WhatsEye provides data-driven insights for deeper understanding.

# Features
✅ Sentiment Analysis – Classifies messages as positive, negative, or neutral using a trained ML model.
✅ Chat Preprocessing – Supports both Android & iOS chat exports for accurate parsing.
✅ User Engagement Analysis – Identifies most active users, ignored users, and response times.
✅ Mood Swing Detection – Detects emotional shifts in conversations over time.
✅ Keyword & Emoji Analysis – Extracts most used words and emojis from chats.
✅ Data Visualizations – Generates interactive charts, word clouds, and heatmaps using Streamlit.
✅ Custom Trained Chatbot – Uses an SVM-based chatbot to answer chat-related queries.

# 🚀 Tech Stack
# Frontend:
Streamlit – Interactive web UI for chat analysis visualization.
# Backend & Processing:
Python – Core programming language for data analysis and machine learning.
Flask / FastAPI – API development (if needed for future enhancements).
# Data Processing & NLP:
NLTK & SpaCy – Sentiment analysis, text preprocessing.
Pandas & NumPy – Data handling and manipulation.
Scikit-learn – Machine learning models (SVM, Logistic Regression).
# Visualization:
Matplotlib & Seaborn – Static plots for trends.
Plotly – Interactive charts for engagement & sentiment.
# Machine Learning & Model Storage:
SVM (Support Vector Machine) – Used for sentiment analysis & chatbot intent classification.
TF-IDF Vectorization – Converts chat messages into numerical features for ML models.
SMOTE (Synthetic Minority Oversampling Technique) – Balances imbalanced sentiment data.
Joblib – Saves & loads trained models efficiently.

# 🎯 How to Use
1️⃣ Export WhatsApp Chat (without media) from Android or iPhone.
2️⃣ Upload the .txt file in WhatsEye’s web interface.
3️⃣ Select the chat type (Android/iOS).
4️⃣ Get instant insights on sentiment, user activity, and patterns.
5️⃣ Use the chatbot to answer queries related to chat analytics.

# 📊 Key Functionalities
1️⃣ Chat Preprocessing
Android & iPhone parsers (android_preprocessor.py, iphone_preprocessor.py) process raw chat exports into structured DataFrames.
2️⃣ Sentiment Analysis (ML Model)
train_model.py trains an SVM-based sentiment classifier on chat data.
Uses TF-IDF vectorization and oversampling (SMOTE) for balancing data.
Outputs labels: Positive, Negative, Neutral.
3️⃣ Chatbot Integration
train_chatbot.py builds an intent-based chatbot trained on chat-specific queries.
Uses SVM classifier to predict intent and respond accordingly.
4️⃣ Data Visualization (Streamlit UI - app.py)
Sentiment Pie Charts (Positive vs. Negative vs. Neutral).
Monthly & Daily Trends (Activity timeline).
Most Active Users & Ignored Messages.
Word Cloud & Emoji Analysis.
Mood Swings Over Time (Emotional variations).

# 📂 Folder Structure

📦 WhatsEye
 ┣ 📜 android_preprocessor.py    # Parses Android chat exports
 ┣ 📜 iphone_preprocessor.py     # Parses iPhone chat exports
 ┣ 📜 train_model.py             # Trains the sentiment analysis model
 ┣ 📜 train_chatbot.py           # Trains the chatbot for query responses
 ┣ 📜 helping.py                 # Contains utility functions for data processing
 ┣ 📜 app.py                     # Main Streamlit app for UI & analytics
 ┗ 📜 README.md                  # Project documentation

# 🔬 Future Enhancements
✅ Multilingual Support – Expand to more languages beyond English & Roman Urdu.
✅ WhatsApp Voice & Media Analysis – Support audio transcription & image analysis.
✅ Cloud Deployment – Deploy as a web app with Firebase/AWS integration.
✅ Advanced NLP Models – Improve accuracy with transformers (BERT, GPT-based models).

🚀 Ready to analyze your WhatsApp chats like never before? Give WhatsEye a try today!




