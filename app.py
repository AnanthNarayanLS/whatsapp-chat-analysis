
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


    #chat Summary

    if st.sidebar.button("Generate Summary"):
        # Filter messages within the selected date range (convert dates appropriately)
        filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        
        if not filtered_df.empty:
            chat_text = " ".join(filtered_df['message'])
            # Remove media and deleted message markers
            chat_text = chat_text.replace("<Media omitted>", "").replace("This message was deleted", "")

            # Summarization using TextRank
            parser = PlaintextParser.from_string(chat_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 5)  # Generate 5 summary sentences
            summarized_sentences = [str(sentence) for sentence in summary]

            # Extract key topics using RAKE
            rake = Rake()
            rake.extract_keywords_from_text(chat_text)
            keywords = [
                kw for kw in rake.get_ranked_phrases()
                if len(kw.split()) > 1 and not any(char in kw for char in "ಠನಡ")
            ][:5]

            # Extract links (and remove duplicates)
            links = [msg for msg in filtered_df['message'] if "http" in msg]
            unique_links = list(set(links))

            # Sentiment Analysis for summary statistics
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(msg)['compound'] for msg in filtered_df['message']]
            positive = sum(1 for score in sentiment_scores if score > 0.2)
            negative = sum(1 for score in sentiment_scores if score < -0.2)
            neutral = len(sentiment_scores) - positive - negative

            # Format fun messages (casual talks) with quotes
            fun_messages = [msg for msg in filtered_df['message'] if "😂" in msg or "🤣" in msg][:5]
            formatted_fun_talks = "\n".join(f"- \"{msg}\"" for msg in fun_messages)

            # Format important conversations with a fixed set of icons
            important_conversations = filtered_df[['date', 'message']].values.tolist()[:5]
            conversation_icons = ["📅", "📍", "💡", "🏡", "😟"]
            formatted_conversations_list = []
            for i, (date, msg) in enumerate(important_conversations):
                icon = conversation_icons[i] if i < len(conversation_icons) else "📅"
                time_str = date.strftime('%I:%M %p') if hasattr(date, 'strftime') else str(date)
                formatted_conversations_list.append(f"- {icon} *[{time_str}]* \"{msg}\"")
            formatted_conversations = "\n".join(formatted_conversations_list)

            # Format shared links with hard-coded labels for demonstration
            formatted_links = ""
            for link in unique_links[:3]:
                if "youtube" in link.lower():
                    formatted_links += f"- 🎓 **YouTube Video:** [Blockchain Course]({link})\n"
                elif "instagram" in link.lower():
                    formatted_links += f"- 🍕 **Instagram Reels:** [Food Vlog]({link})\n"
                elif "maps" in link.lower():
                    formatted_links += f"- 🗺️ **Google Maps:** [Meet-up Location]({link})\n"
                else:
                    formatted_links += f"- 🔗 [External Link]({link})\n"

            # Build the final formatted summary following the new output format
            formatted_summary = f"""📌 **Key Topics Discussed**
            {chr(10).join(f"- {topic}" for topic in keywords)}

            🗣 **Important Conversations**
            {formatted_conversations}

            😂 **Casual & Fun Talks**
            {formatted_fun_talks}

            🔗 **Shared Links**
            {formatted_links}
            📢 **Sentiment Summary**
            - ✅ **Positive Chat:** {round((positive/len(sentiment_scores))*100)}% (Casual fun, celebration, study plans)  
            - ❌ **Negative Chat:** {round((negative/len(sentiment_scores))*100)}% (Lost item, minor conflicts)  
            - ➖ **Neutral Chat:** {round((neutral/len(sentiment_scores))*100)}%  

            📊 **Overall Mood:** {"**Positive & Friendly 🎉**" if positive > negative else "**Mixed / Slightly Negative 😐**"}"""

            # Display the formatted summary in Streamlit
            st.title(":blue[Chat Summary]")
            st.markdown(formatted_summary, unsafe_allow_html=True)

            # Optional: Print summary to the console for debugging
            print("\n=== Chat Summary ===")
            print(formatted_summary)
        else:
            st.write(":red[No messages found in the selected date range]")
