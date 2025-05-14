import pandas as pd
import networkx as nx
import time
import matplotlib.pyplot as plt
from itertools import islice
from networkx.algorithms.community import girvan_newman, modularity
import community as community_louvain  # pip install python-louvain

# Load graph
tiktok_df = pd.read_csv("tiktok_data.csv")
drugs = ['Upadacitinib', 'Rinvoq', 'Jak inhibitors', 'Abrocitinib', 'Cibinqo', 'Baricitinib', 'Olumiant',
         'Ritlecitinib', 'Litfulo', 'Tofacitinib', 'Xeljanz', 'Filgotinib', 'Jyseleca', 'Deucravacitinib',
         'Sotyktu', 'Delgocitinib', 'Corectim', 'Ruxolitinib', 'Jakavi', 'Opzelura', 'Peficitinib', 'Smyraf']

# Rebuild graph
G = nx.Graph()
for _, row in tiktok_df.iterrows():
    post_id = str(row['post_id']) if pd.notnull(row['post_id']) else None
    user_id = row['author_unique_id'] if pd.notnull(row['author_unique_id']) else "Unknown_User"
    combined_text = ' '.join([str(row.get(f, '')) for f in ['translated_desc', 'translated_video_text', 'translated_text']])

    if not post_id or not combined_text.strip():
        continue

    G.add_node(post_id, type='video')
    G.add_node(user_id, type='user')
    G.add_edge(user_id, post_id, type='user_to_video')

    for drug in drugs:
        if drug.lower() in combined_text.lower():
            G.add_node(drug, type='drug')
            G.add_edge(post_id, drug, type='video_to_drug')

# --- Louvain Method ---
start = time.time()
louvain_partition = community_louvain.best_partition(G)
louvain_communities = {}
for node, comm_id in louvain_partition.items():
    louvain_communities.setdefault(comm_id, []).append(node)
louvain_modularity = modularity(G, list(louvain_communities.values()))
louvain_time = time.time() - start

# --- Girvan–Newman Method (Top 5 splits only) ---
start = time.time()
gn_communities_gen = girvan_newman(G)
gn_communities = list(islice(gn_communities_gen, 5))[-1]
gn_modularity = modularity(G, gn_communities)
gn_time = time.time() - start

# --- Conductance (simplified average of cuts between communities) ---
def average_conductance(G, communities):
    conductances = []
    for community in communities:
        S = set(community)
        cut_size = len(list(nx.edge_boundary(G, S)))
        volume = sum(dict(G.degree(S)).values())
        if volume > 0:
            conductance = cut_size / volume
            conductances.append(conductance)
    return sum(conductances) / len(conductances)


louvain_conductance = average_conductance(G, list(louvain_communities.values()))
gn_conductance = average_conductance(G, gn_communities)

# --- Comparison Report ---
print("=== Community Detection Comparison ===")
print(f"Louvain:     Communities = {len(louvain_communities):3d} | Modularity = {louvain_modularity:.4f} | Avg Conductance = {louvain_conductance:.4f} | Time = {louvain_time:.2f}s")
print(f"Girvan-Newman: Communities = {len(gn_communities):3d} | Modularity = {gn_modularity:.4f} | Avg Conductance = {gn_conductance:.4f} | Time = {gn_time:.2f}s")

# --- Optional: Bar Chart ---
metrics_df = pd.DataFrame({
    'Method': ['Louvain', 'Girvan–Newman'],
    'Modularity': [louvain_modularity, gn_modularity],
    'Conductance': [louvain_conductance, gn_conductance],
    'Time (s)': [louvain_time, gn_time],
    'Communities': [len(louvain_communities), len(gn_communities)]
})

metrics_df.set_index('Method')[['Modularity', 'Conductance', 'Time (s)', 'Communities']].plot(kind='bar', figsize=(10, 6), rot=0)
plt.title("Comparison of Community Detection Methods")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.show()
