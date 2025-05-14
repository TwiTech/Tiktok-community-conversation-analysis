import pandas as pd
from openai import OpenAI
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Set your OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
df_tiktok = pd.read_csv("tiktok_data.csv")
df_community = pd.read_csv("tiktok_community_nodes.csv")

# Top 5 Louvain community IDs (as identified earlier)
top5_ids = [0, 217, 115, 69, 16]

# Collect text data
samples = defaultdict(list)

for cid in top5_ids:
    community_nodes = df_community[
        (df_community["Louvain Community"] == cid) & (df_community["Type"] == "video")
    ]["Node"].astype(str).tolist()

    matched = df_tiktok[df_tiktok["post_id"].astype(str).isin(community_nodes)]

    for _, row in matched.iterrows():
        combined_text = ' '.join([
            str(row.get('translated_desc', '')),
            str(row.get('translated_video_text', '')),
            str(row.get('translated_text', ''))
        ]).strip()
        if combined_text:
            samples[cid].append(combined_text)

# LLM Prompt Function
def analyze_experiences(texts, max_chars=6000):
    combined = "\n\n".join(texts)[:max_chars]
    prompt = f"""You are a medical analyst reviewing patient-generated content about JAK inhibitors.

Based on the following posts, provide:
1. A high-level summary of the patient experiences.
2. Common themes (positive and negative).
3. Insights on emotional tone or concerns.

Text:
{combined}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Run analysis per community
print("==== Patient Experience Analysis per Community ====")
for cid in top5_ids:
    print(f"\n--- Louvain Community {cid} ---")
    result = analyze_experiences(samples[cid][:30])  # limit to 30 posts per community
    print(result)
