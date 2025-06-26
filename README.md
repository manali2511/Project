#Code to Upload File
from google.colab import files
uploaded = files.upload()

#Then Load It
import pandas as pd

df = pd.read_csv('Trending videos on youtube dataset.csv')
df.head()

#Clean Nulls and Format Columns

# Drop rows with missing essential values
df.dropna(subset=['videoTitle', 'viewCount', 'videoCategoryLabel'], inplace=True)

# Convert publish date to datetime
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# Standardize category names if needed
df['videoCategoryLabel'] = df['videoCategoryLabel'].str.strip().str.title()

#Sentiment Analysis on Titles & Tags
#Install and Apply TextBlob

pip install textblob

from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['title_sentiment'] = df['videoTitle'].apply(get_sentiment)
df['desc_sentiment'] = df['videoDescription'].apply(get_sentiment)

#SQL Query
import sqlite3

conn = sqlite3.connect("trending.db")
df.to_sql("trending_data", conn, if_exists="replace", index=False)

#Time-Series Analysis – Trending Duration
#Identify Trending Duration (if video appears multiple times)
# Count how many unique days each video appeared
df['date_only'] = df['publishedAt'].dt.date
duration_df = df.groupby('videoId')['date_only'].nunique().reset_index()
duration_df.columns = ['videoId', 'trending_days']
#Merge Back & Visualize
df = df.merge(duration_df, on='videoId', how='left')

#Visualizations with Python (Seaborn/Matplotlib)
#Top Categories by Views

import seaborn as sns
import matplotlib.pyplot as plt

top_categories = df.groupby('videoCategoryLabel')['viewCount'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_categories.values, y=top_categories.index)
plt.title("Top 10 Video Categories by Avg Views")
plt.xlabel("Average Views")
plt.show()

#Sentiment Distribution
sns.histplot(df['title_sentiment'], kde=True, bins=20)
plt.title("Sentiment Distribution of Video Titles")
plt.show()

#Trending Duration Distribution
sns.histplot(df['trending_days'], bins=30)
plt.title("Distribution of Video Trending Duration")
plt.xlabel("Number of Trending Days")
plt.show()

#PYTHON DASHBOARD CODE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv("Trending videos on youtube dataset.csv")

# Clean data
df.dropna(subset=['videoTitle', 'viewCount', 'videoCategoryLabel'], inplace=True)
df['videoCategoryLabel'] = df['videoCategoryLabel'].str.strip().str.title()

# Sentiment analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['title_sentiment'] = df['videoTitle'].apply(get_sentiment)

def sentiment_label(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_category'] = df['title_sentiment'].apply(sentiment_label)

# Visual 1: Top 10 Genres by Avg Views
popular_genres = df.groupby('videoCategoryLabel')['viewCount'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=popular_genres.values, y=popular_genres.index, palette="viridis")
plt.title("Top 10 Most Popular YouTube Genres by Avg Views")
plt.xlabel("Average Views")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# Visual 2: Sentiment Distribution
sentiment_distribution = df['sentiment_category'].value_counts()
plt.figure(figsize=(6, 6))
sentiment_distribution.plot.pie(autopct='%1.1f%%', startangle=140, colors=['#FFDD57', '#57D9A3', '#FF6B6B'])
plt.title("Sentiment Distribution of Video Titles")
plt.ylabel('')
plt.tight_layout()
plt.show()
