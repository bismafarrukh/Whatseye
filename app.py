from helping import get_chatbot_response
import streamlit as st
import android_preprocessor
import iphone_preprocessor
import helping
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from helping import integrate_sentiment_analysis, find_most_ignored_person, response_time_analysis, detect_mood_swings

st.sidebar.markdown(
    """
    <style>
        /* File Uploader Box */
        div[data-testid="stFileUploader"] {
            border: 2px solid #A8E6B8 !important; /* Soft Light Green Border */
            background-color: #F8F9FA !important; /* Light Background */
            color: #146B3A !important; /* Deep Green Text */
            text-align: center;
            padding: 10px;
            border-radius: 8px;
            transition: 0.3s;
        }

        /* File Uploader Hover Effect */
        div[data-testid="stFileUploader"]:hover {
            background-color: #E3F2E3 !important; /* Slightly Darker Light Green */
        }

        /* Sidebar Container */
        .sidebar {
            background-color: #E0F2E9; /* Light Green */
            border: 2px solid #1EBB5F !important; /* WhatsApp Green Border */
            background: linear-gradient(135deg, #A8E6B8, #D4ECDD); /* Soft Green Gradient */
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }

        /* Sidebar Title */
        .sidebar-title {
            color: #146B3A;  /* Deep Green */
            font-size: 40px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
    </style>

    <div>
        <div class="sidebar">
            <p class="sidebar-title"> WHATSEYE </p>    
        </div> 
    </div>
    """,
    unsafe_allow_html=True
)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file", key="Whatsapp_File")




if uploaded_file is not None:
    try:
        # Read and decode uploaded file
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode('utf-8')

        # Select chat type
        chat_type = st.sidebar.radio("Select chat type:", ("Android", "iOS"))

        # Process data based on selected chat type
        if chat_type == "Android":
            df = android_preprocessor.android_preprocessor(data)
            st.title("Analysis for Android chats:")
        elif chat_type == "iOS":
            df = iphone_preprocessor.iphone_preprocessor(data)
            st.title("Analysis for iOS chats:")

        # Ensure DataFrame is valid
        if df.empty:
            st.error("The processed data is empty. Please check your uploaded file.")
        else:
           # Fetch unique users while removing group notifications
            user_list = df[~df['user'].str.contains("group_notification|removed|changed the group|left", na=False)]['user'].unique().tolist()
            user_list.sort()
            user_list.insert(0, "Overall")  # Add "Overall" option

            # Sidebar selection
            selected_user = st.sidebar.selectbox("Show analysis for:", user_list)

            # Perform sentiment analysis before visualizing
            df = integrate_sentiment_analysis(df)

            # Get sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()

            if sentiment_counts is None:
                st.warning("No messages found for the selected user.")
            else:
                color_map = {'Negative': 'red', 'Positive': 'green', 'Neutral': 'gray'}
                st.title("Sentiment Analysis Results")
                fig = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values,
                             title=f"Sentiment Distribution for {selected_user}", color=sentiment_counts.index, 
                            color_discrete_map=color_map)
                st.plotly_chart(fig)
            

            st.title("Sentiment Trend Over Time")

            # Perform sentiment analysis
            df = integrate_sentiment_analysis(df)


            # Ensure all months are included (short form)
            all_months_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            # Ensure all months are included (full form)
            all_months_full = ["January", "February", "March", "April", "May", "June", "July", "August",
                            "September", "October", "November", "December"]

            # Convert full month names to short form
            df['month'] = df['month'].replace(dict(zip(all_months_full, all_months_short)))

            # Convert month column to categorical for proper ordering
            df['month'] = pd.Categorical(df['month'], categories=all_months_short, ordered=True)

            # Group by month & sentiment
            sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

            # Ensure all months are present
            sentiment_trend = sentiment_trend.reindex(index=all_months_short, fill_value=0)

            # Create the stacked bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sentiment_trend.index,
                y=sentiment_trend.get('Positive', 0),
                name='Positive',
                marker=dict(color='lightgreen')
            ))

            fig.add_trace(go.Bar(
                x=sentiment_trend.index,
                y=-sentiment_trend.get('Negative', 0),  # Negative values for proper stacking
                name='Negative',
                marker=dict(color='lightcoral')
            ))

            # Add line graph for total messages
            total_messages = sentiment_trend.sum(axis=1)  # Sum of all sentiment categories
            fig.add_trace(go.Scatter(
                x=sentiment_trend.index,
                y=total_messages,
                mode='lines+markers',
                name='Total Messages',
                line=dict(color='black', width=2)
            ))

            # Layout settings
            fig.update_layout(
                barmode='relative',
                title="Sentiment Trend Chart",
                xaxis_title="Months",
                yaxis_title="Satisfied (↑) / Dissatisfied (↓)",
                legend_title="Sentiment",
                template="plotly_white"
            )

            # Display the plot
            st.plotly_chart(fig)

            # Fetch statistics
            num_messages, words, num_media_messages, num_links = helping.fetch_stats(selected_user, df)
            st.title("Top Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Total Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Total Links Shared")
                st.title(num_links)

            # Monthly Timeline
            st.title("Monthly Timeline")
            timeline = helping.monthly_timeline(selected_user, df)
            if timeline.empty:
                st.warning("No data available for the monthly timeline.")
            else:
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Daily Timeline
            st.title("Daily Timeline")
            daily_timeline = helping.daily_timeline(selected_user, df)
            if daily_timeline.empty:
                st.warning("No data available for the daily timeline.")
            else:
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Perform analysis
                ignored_users = find_most_ignored_person(df)
                response_analysis = response_time_analysis(df)
                mood_swings = detect_mood_swings(df)

                # Display results
                st.title("Chat Analysis Dashboard")

                st.subheader("Most Ignored Users")
                st.dataframe(ignored_users)

                st.subheader("Response Time Analysis")
                st.dataframe(response_analysis)

            if mood_swings.empty:
                st.warning("No significant mood swings detected.")
            else:
                # Create a line graph of mood swings over time
                fig = px.line(mood_swings, 
                            x="date", 
                            y="sentiment_numeric", 
                            color="user", 
                            markers=True, 
                            title="Mood Swings Over Time",
                            labels={"sentiment_numeric": "Sentiment Score", "date": "Time"},
                            line_shape="hv")  # Step-like changes

                st.plotly_chart(fig)

            # Activity Map
            st.title("Activity Map")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most Busy Day")
                busy_day = helping.week_activity_map(selected_user, df)
                if busy_day.empty:
                    st.warning("No data available for the most busy day.")
                else:
                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            with col2:
                st.header("Most Busy Month")
                busy_month = helping.month_activity_map(selected_user, df)
                if busy_month.empty:
                    st.warning("No data available for the most busy month.")
                else:
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # Weekly Activity Heatmap
            st.title("Weekly Activity Map")
            user_heatmap = helping.activity_heatmap(selected_user, df)
            if user_heatmap.empty:
                st.warning("No data available for the weekly activity heatmap.")
            else:
                fig, ax = plt.subplots()
                sns.heatmap(user_heatmap, ax=ax, cmap="coolwarm")
                st.pyplot(fig)

            # Most Busy Users
            if selected_user == "Overall":
                st.title("Most Busy Users")
                x, new_df = helping.most_busy_users(df)
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='skyblue')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # Word Cloud
            st.title("Word Cloud")
            df_wc = helping.create_wordcloud(selected_user, df)
            if df_wc is None:
                st.warning("No data available for the word cloud.")
            else:
                fig, ax = plt.subplots()
                ax.imshow(df_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            # Most Common Words
            st.title("Most Common Words")
            most_common_df = helping.most_common_words(selected_user, df)
            if most_common_df.empty:
                st.warning("No data available for most common words.")
            else:
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='teal')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Emoji Analysis
            st.title("Emoji Analysis")
            emoji_df = helping.emoji_helper(selected_user, df)
            if emoji_df.empty:
                st.warning("No data available for emoji analysis.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig = px.pie(
                        emoji_df.head(),
                        names='emoji',
                        values='count',
                        title="Emoji Distribution"
                    )
                    st.plotly_chart(fig)
            st.title("WhatsApp Chat Analysis Bot 🤖")

            user_input = st.text_input("Ask me about the chat analysis...")

            if user_input:
                response = get_chatbot_response(user_input)  # Call the helper function
                st.write(response)  # Display chatbot response

                        
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.warning("Please upload a file to continue.")
