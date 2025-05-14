import pandas as pd
import networkx as nx
from openai import OpenAI
import os
from collections import defaultdict
from dotenv import load_dotenv

# --- Set your API key securely ---
load_dotenv()  # Load environment variables from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# --- Load files ---
df_tiktok = pd.read_csv("tiktok_data.csv")
df_community = pd.read_csv("tiktok_community_nodes.csv")

# --- Identify Top 5 Louvain communities by number of videos ---
top5_ids = df_community[df_community['Type'] == 'video']['Louvain Community'].value_counts().head(5).index.tolist()

# --- Build a mapping from community to their video text content ---
community_texts = defaultdict(list)

for _, row in df_tiktok.iterrows():
    post_id = str(row['post_id'])
    content = ' '.join([str(row.get(c, '')) for c in ['translated_desc', 'translated_video_text', 'translated_text']]).strip()
    if not content:
        continue

    comm_row = df_community[df_community['Node'] == post_id]
    if not comm_row.empty:
        community_id = comm_row['Louvain Community'].values[0]
        if community_id in top5_ids:
            community_texts[community_id].append(content)

# --- Label communities using GPT ---
def label_community_text(texts, max_chars=6000):
    combined = "\n\n".join(texts)[:max_chars]
    prompt = f"""You are analyzing a community of TikTok posts discussing medications. Based on the following user content, generate:
1. A short label (max 4 words)
2. A one-sentence summary of what this community is talking about.

Content:
{combined}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- Label and print results ---
print("==== LLM-Labeled Communities ====")
for cid in top5_ids:
    print(f"\nLouvain Community {cid}")
    output = label_community_text(community_texts[cid])
    print(output)
