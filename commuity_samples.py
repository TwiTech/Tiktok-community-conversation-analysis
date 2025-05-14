# Re-import after kernel reset
import pandas as pd
from collections import defaultdict

# Reload data
df_tiktok = pd.read_csv("tiktok_data.csv")
df_community = pd.read_csv("tiktok_community_nodes.csv")

# Define top 5 Louvain community IDs
top5_louvain_ids = [0, 217, 115, 69, 16]

# Collect 3 sample posts per community
samples = defaultdict(list)

for cid in top5_louvain_ids:
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
        if combined_text and len(samples[cid]) < 3:
            samples[cid].append(combined_text)

