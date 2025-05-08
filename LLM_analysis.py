import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
from openai import OpenAI
import getpass
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Securely obtain OpenAI API key
# def get_api_key():
#     api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     api_key = getpass.getpass("Enter your OpenAI API key: ")
    # return api_key

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Data Preprocessing
def load_and_clean_data(file_path):
    """Load and clean TikTok dataset."""
    try:
        df = pd.read_csv(file_path, parse_dates=['createTime'], low_memory=False)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    # Filter posts mentioning JAK inhibitors
    keywords = r'rivoq|upadacitinib|jak inhibitor'
    df['relevant'] = df[['translated_desc', 'translated_video_text', 'translated_text']].apply(
        lambda x: x.str.contains(keywords, case=False, na=False).any(), axis=1
    )
    df = df[df['relevant']].drop(columns=['relevant'])

    # Clean text data
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize whitespace and case
        return text

    df['translated_desc'] = df['translated_desc'].apply(clean_text)
    df['translated_video_text'] = df['translated_video_text'].apply(clean_text)
    df['translated_text'] = df['translated_text'].apply(clean_text)

    # Parse stats JSON
    def parse_stats(stats):
        try:
            stats_dict = json.loads(stats) if isinstance(stats, str) else {}
            return {
                'diggCount': stats_dict.get('diggCount', 0),
                'shareCount': stats_dict.get('shareCount', 0),
                'commentCount': stats_dict.get('commentCount', 0),
                'playCount': stats_dict.get('playCount', 0),
                'collectCount': stats_dict.get('collectCount', 0)
            }
        except json.JSONDecodeError:
            return {'diggCount': 0, 'shareCount': 0, 'commentCount': 0, 'playCount': 0, 'collectCount': 0}

    stats_df = df['stats'].apply(parse_stats).apply(pd.Series)
    df = pd.concat([df.drop(columns=['stats']), stats_df], axis=1)

    # Handle missing data
    df = df.dropna(subset=['translated_video_text', 'translated_text'], how='all')
    df = df.fillna({'diggCount': 0, 'shareCount': 0, 'commentCount': 0, 'playCount': 0, 'collectCount': 0})

    return df

# Step 2: GPT-4-Based Text Analysis
def analyze_sentiment(text):
    """Perform sentiment analysis using GPT-4."""
    if not text or len(text.split()) < 3:
        return "Neutral", 0.5
    prompt = f"""
    Classify the sentiment expressed in the following text about JAK inhibitors as Positive, Negative, or Neutral.
    Provide a brief explanation for the classification.
    Text: {text[:1000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        # Parse response (assuming GPT-4 returns sentiment and explanation)
        if "Positive" in result:
            return "Positive", 0.9
        elif "Negative" in result:
            return "Negative", 0.9
        else:
            return "Neutral", 0.5
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Neutral", 0.5

def extract_themes(texts, num_themes=5):
    """Extract themes using GPT-4."""
    combined_text = " ".join([t for t in texts if t])[:4000]  # Limit to avoid token overflow
    prompt = f"""
    Identify the main themes or topics discussed in the following text about JAK inhibitors.
    List up to {num_themes} themes (e.g., side effects, efficacy, emotional impact) and provide a short description for each.
    Text: {combined_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
        # Parse themes (assuming list format)
        themes = []
        lines = result.split('\n')
        for line in lines[:num_themes]:
            if line.strip():
                parts = line.split(':', 1)
                if len(parts) == 2:
                    themes.append({'theme': parts[0].strip(), 'description': parts[1].strip()})
        return themes
    except Exception as e:
        print(f"Error in theme extraction: {e}")
        return [{"theme": "Unknown", "description": "Error in processing"}]

def summarize_narrative(text, post_id):
    """Generate narrative summary using GPT-4."""
    if not text or len(text.split()) < 10:
        return f"No sufficient content for summary (post_id: {post_id})."
    prompt = f"""
    Summarize the patient experience with JAK inhibitors described in the following text in 2-3 sentences,
    focusing on their feelings, outcomes, and challenges.
    Text: {text[:1000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in narrative summarization (post_id: {post_id}): {e}")
        return f"Error in summarization (post_id: {post_id})."

# Step 3: Quantitative Analysis
def quantitative_analysis(df):
    """Perform engagement and correlation analysis."""
    # Normalize engagement metrics by post age
    df['post_age_days'] = (datetime.now() - df['createTime']).dt.days
    df['digg_per_day'] = df['diggCount'] / df['post_age_days'].replace(0, 1)
    df['play_per_day'] = df['playCount'] / df['post_age_days'].replace(0, 1)

    # Correlation analysis
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    corr_digg, p_digg = spearmanr(df['sentiment_score'], df['digg_per_day'], nan_policy='omit')
    corr_play, p_play = spearmanr(df['sentiment_score'], df['play_per_day'], nan_policy='omit')

    # Temporal trends
    df['week'] = df['createTime'].dt.to_period('W')
    temporal_data = df.groupby('week').agg({
        'sentiment': lambda x: x.value_counts().to_dict(),
        'playCount': 'sum'
    }).reset_index()

    # Author analysis
    author_stats = df.groupby('author_unique_id').agg({
        'post_id': 'count',
        'diggCount': 'mean',
        'playCount': 'mean',
        'sentiment': lambda x: x.value_counts().to_dict()
    }).reset_index().rename(columns={'post_id': 'post_count'})

    return {
        'correlations': {'digg': (corr_digg, p_digg), 'play': (corr_play, p_play)},
        'temporal': temporal_data,
        'author_stats': author_stats
    }

# Step 4: Visualization
def create_visualizations(df, quant_results, output_dir="plots"):
    """Generate and save visualizations."""
    plt.style.use('seaborn')
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Distribution of TikTok Posts')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Posts')
    plt.savefig(f"{output_dir}/sentiment_distribution.png")
    plt.close()

    # Engagement vs. Sentiment
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sentiment', y='diggCount', hue='sentiment', size='playCount')
    plt.title('Engagement (Likes) vs. Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Likes (Digg Count)')
    plt.savefig(f"{output_dir}/engagement_vs_sentiment.png")
    plt.close()

    # Temporal Trends
    temporal = quant_results['temporal']
    plt.figure(figsize=(10, 6))
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        counts = [d.get(sentiment, 0) for d in temporal['sentiment']]
        plt.plot(temporal['week'].astype(str), counts, label=sentiment)
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Week')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_trends.png")
    plt.close()

    # Interesting Fact: Sentiment Proportion
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Proportions of Rivoq Posts')
    plt.savefig(f"{output_dir}/sentiment_proportions.png")
    plt.close()

# Main Function
def main():
    file_path = "tiktok_data.csv"
    df = load_and_clean_data(file_path)
    if df is None:
        return

    # Step 2: Text Analysis with GPT-4
    print("Performing sentiment analysis...")
    df['sentiment'], df['sentiment_score'] = zip(*df['translated_video_text'].apply(analyze_sentiment))
    
    print("Extracting themes...")
    texts = (df['translated_video_text'] + ' ' + df['translated_desc'] + ' ' + df['translated_text']).tolist()
    themes = extract_themes(texts)
    
    print("Generating narrative summaries...")
    df['narrative_summary'] = df.apply(
        lambda row: summarize_narrative(row['translated_video_text'], row['post_id']), axis=1
    )

    # Step 3: Quantitative Analysis
    print("Performing quantitative analysis...")
    quant_results = quantitative_analysis(df)

    # Step 4: Visualization
    print("Generating visualizations...")
    create_visualizations(df, quant_results)

    # Step 5: Save Results
    print("Saving results...")
    df.to_csv("processed_tiktok_data.csv", index=False)
    with open("themes.json", "w") as f:
        json.dump(themes, f, indent=2)
    with open("quantitative_results.json", "w") as f:
        json.dump({
            'correlations': quant_results['correlations'],
            'author_stats': quant_results['author_stats'].to_dict(orient='records')
        }, f, indent=2)
    df[['post_id', 'narrative_summary']].to_csv("narrative_summaries.csv", index=False)

    # Step 6: Print Summary
    print("\nAnalysis Complete!")
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    print("\nTop Themes:")
    for theme in themes:
        print(f"- {theme['theme']}: {theme['description']}")
    print("\nCorrelation Results:")
    print(f"Digg vs. Sentiment: rho={quant_results['correlations']['digg'][0]:.2f}, p={quant_results['correlations']['digg'][1]:.2f}")
    print(f"Play vs. Sentiment: rho={quant_results['correlations']['play'][0]:.2f}, p={quant_results['correlations']['play'][1]:.2f}")
    print("\nInfluential Authors:")
    print(quant_results['author_stats'][['author_unique_id', 'post_count', 'diggCount', 'playCount']].head())
    print("\nSample Narrative Summaries:")
    print(df[['post_id', 'narrative_summary']].head())

if __name__ == "__main__":
    main()