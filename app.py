'''
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer  # Improved TextRank summarization
import nltk
from rake_nltk import Rake  # For extracting key topics

st.sidebar.title("WhatsApp Chat Analyzer")

#create a file uploader to upload txt file
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    #call the preprocess func to give df
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Add Date Selection for Summarization
    st.sidebar.header("Summarize Chat in Date Range")
    start_date = st.sidebar.date_input("Start Date", df['date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['date'].max().date())

    if st.sidebar.button("Show Analysis"):
        # Display basic stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header(":blue[Total Messages]")
            st.title(num_messages)
        with col2:
            st.header(":blue[Total Words]")
            st.title(words)
        with col3:
            st.header(":blue[Media Shared]")
            st.title(num_media_messages)
        with col4:
            st.header(":blue[Links Shared]")
            st.title(num_links)

        # Monthly timeline
        st.title(":blue[Monthly Chat Timeline]")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title(":blue[Daily Timeline]")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title(':blue[Activity Map]')
        col1, col2 = st.columns(2)
        # Weekly activity
        with col1:
            st.header(":green[Most busy day]")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Monthly activity
        with col2:
            st.header(":green[Most busy month]")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Heatmap
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Finding the busiest users (Group level)
        if selected_user == 'Overall':
            st.title(':blue[Most Busy Users]')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud (Top Frequent Words)
        st.title(":blue[Wordcloud]")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        st.title(':blue[Most common words]')
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji Analysis
        st.title(":blue[Emoji Analysis]")
        emoji_df = helper.emoji_helper(selected_user, df)
        if emoji_df.shape[0] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)
        else:
            st.write(":red[No Emojis Sent by this user]")

    # Summarization Section
    if st.sidebar.button("Generate Summary"):
        filtered_df = df[(df['date'] >= str(start_date)) & (df['date'] <= str(end_date))]

        if not filtered_df.empty:
            chat_text = " ".join(filtered_df['message'])

            # Remove media messages
            chat_text = chat_text.replace("<Media omitted>", "")

            # Summarization using TextRank
            parser = PlaintextParser.from_string(chat_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 5)  # Generate 5 summary sentences
            
            summarized_sentences = [str(sentence) for sentence in summary]

            # Extract keywords using RAKE
            rake = Rake()
            rake.extract_keywords_from_text(chat_text)
            keywords = rake.get_ranked_phrases()[:5]  # Get top 5 keywords

            # Format Summary
            formatted_summary = "**📌 Key Topics:**\n" + ", ".join(keywords) + "\n\n**📝 Chat Summary:**\n" + "\n".join(f"- {sentence}" for sentence in summarized_sentences)

            # Display Summary
            st.title(":blue[Chat Summary]")
            st.markdown(formatted_summary)

            # Print Summary to CMD for Debugging
            print("\n=== Chat Summary ===")
            print(formatted_summary)

        else:
            st.write(":red[No messages found in the selected date range]")

'''
import os
import pandas as pd
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer  # Improved TextRank summarization
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from rake_nltk import Rake  # For extracting key topics
from nltk.sentiment import SentimentIntensityAnalyzer  # VADER for emotion detection

# Ensure the correct path for NLTK data
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Force-download necessary tokenizers
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)  # Fix for missing `punkt_tab`




# Download required NLTK data
nltk.download('vader_lexicon')

st.sidebar.title("WhatsApp Chat Analyzer")

# Upload file
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Call the preprocess function
    df = preprocessor.preprocess(data)

    print(df.head(5))

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Add Date Selection for Summarization
    st.sidebar.header("Summarize Chat in Date Range")
    start_date = st.sidebar.date_input("Start Date", df['date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['date'].max().date())

    if st.sidebar.button("Show Analysis"):
        # Display basic stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header(":blue[Total Messages]")
            st.title(num_messages)
        with col2:
            st.header(":blue[Total Words]")
            st.title(words)
        with col3:
            st.header(":blue[Media Shared]")
            st.title(num_media_messages)
        with col4:
            st.header(":blue[Links Shared]")
            st.title(num_links)

        # Monthly timeline
        st.title(":blue[Monthly Chat Timeline]")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title(":blue[Daily Timeline]")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title(':blue[Activity Map]')
        col1, col2 = st.columns(2)
        # Weekly activity
        with col1:
            st.header(":green[Most busy day]")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Monthly activity
        with col2:
            st.header(":green[Most busy month]")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Heatmap
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

    # **Emotion Detection for Group Chats**
    if selected_user == "Overall":
        st.title(":blue[Emotion Detection for Group Members]")

        sia = SentimentIntensityAnalyzer()
        user_emotions = {}

        for user in df['user'].unique():
            if user != 'group_notification':
                user_df = df[df['user'] == user]
                scores = user_df['message'].apply(lambda msg: sia.polarity_scores(msg)['compound'])
                avg_score = scores.mean()
                user_emotions[user] = avg_score

        emotion_df = (
            pd.DataFrame(user_emotions.items(), columns=['User', 'Emotion Score'])
            .sort_values(by='Emotion Score', ascending=False)
        )

        fig, ax = plt.subplots()
        sns.barplot(x='User', y='Emotion Score', data=emotion_df, ax=ax, palette="coolwarm")
        plt.xticks(rotation=90)
        st.pyplot(fig)




    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Title
    if selected_user != "Overall":
        st.title(":blue[Relationship Status Prediction]")

        # Sentiment Analysis
        sentiment_scores = df['message'].apply(lambda msg: sia.polarity_scores(msg)['compound'])
        avg_sentiment = sentiment_scores.mean()

        # Categorize Messages Based on Sentiment
        df['sentiment_category'] = df['message'].apply(lambda msg: 
            "Positive" if sia.polarity_scores(msg)['compound'] > 0.2 else 
            "Negative" if sia.polarity_scores(msg)['compound'] < -0.2 else 
            "Neutral"
        )

        # Count Each Sentiment Type
        sentiment_counts = df['sentiment_category'].value_counts()

        # **Plot Emotion Chart**
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['green', 'gray', 'red'], ax=ax)
        ax.set_title("Emotion Distribution in Messages")
        ax.set_xlabel("Sentiment Type")
        ax.set_ylabel("Message Count")

        # Display the Chart in Streamlit
        st.pyplot(fig)

        # **Relationship Status Logic**
        status = "Unknown"  # ✅ Default value to avoid NameError

        if 'date' in df.columns:
            response_times = df['date'].diff().dt.total_seconds().mean() if len(df) > 1 else None
        else:
            response_times = None

        message_lengths = df['message'].apply(len).mean()  # Avg message length
        emoji_count = df['message'].str.count(r'[😀😂😍❤️🔥🥺😭😡]')  # Count emojis

        # Define Relationship Categories
        if avg_sentiment > 0.5 and emoji_count.mean() > 3:
            status = "Best Friends"
        elif avg_sentiment > 0.3 and emoji_count.mean() > 1:
            status = "Close Friends"
        elif avg_sentiment > 0.2 and "❤️" in df['message'].values:
            status = "Romantic / Flirty"
        elif avg_sentiment > 0.2:
            status = "Friendly / Positive"
        elif avg_sentiment < -0.3 and "😡" in df['message'].values:
            status = "Conflicted / Tense"
        elif avg_sentiment < -0.2:
            status = "Tense / Negative"
        elif df.shape[0] < 5:
            status = "Distant / Lost Touch"
        elif response_times and response_times > 86400:
            status = "Ghosting / Ignored"
        elif message_lengths < 5:
            status = "Fake / Forced Interaction"
        elif avg_sentiment > -0.2 and avg_sentiment < 0.2:
            status = "Neutral / Professional"
        elif emoji_count.mean() < 1 and "?" in df['message'].values:
            status = "Informational / Transactional"
        else:
            status = "Unpredictable"

        # Display the Predicted Relationship Status
        st.subheader(f"Predicted Relationship Status: **{status}**")



    # Summarization Section
    if st.sidebar.button("Generate Summary"):
        filtered_df = df[(df['date'] >= str(start_date)) & (df['date'] <= str(end_date))]

        if not filtered_df.empty:
            chat_text = " ".join(filtered_df['message'])

            # Remove media messages
            chat_text = chat_text.replace("<Media omitted>", "")

            # Summarization using TextRank
            parser = PlaintextParser.from_string(chat_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 5)  # Generate 5 summary sentences
            
            summarized_sentences = [str(sentence) for sentence in summary]

            # Extract keywords using RAKE
            rake = Rake()
            rake.extract_keywords_from_text(chat_text)
            keywords = rake.get_ranked_phrases()[:5]  # Get top 5 keywords

            # Format Summary
            formatted_summary = "**📌 Key Topics:**\n" + ", ".join(keywords) + "\n\n**📝 Chat Summary:**\n" + "\n".join(f"- {sentence}" for sentence in summarized_sentences)

            # Display Summary
            st.title(":blue[Chat Summary]")
            st.markdown(formatted_summary)

            # Print Summary to CMD for Debugging
            print("\n=== Chat Summary ===")
            print(formatted_summary)

        else:
            st.write(":red[No messages found in the selected date range]")
