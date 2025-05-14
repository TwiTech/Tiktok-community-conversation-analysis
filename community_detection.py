# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import community as community_louvain  # Louvain algorithm
# from networkx.algorithms.community import girvan_newman
# from itertools import islice

# # Load TikTok data
# tiktok_path = "tiktok_data.csv"
# tiktok_df = pd.read_csv(tiktok_path)

# # Define drug list
# drugs = [
#     'Upadacitinib', 'Rinvoq', 'Jak inhibitors',
#     'Abrocitinib', 'Cibinqo', 'Baricitinib', 'Olumiant',
#     'Ritlecitinib', 'Litfulo', 'Tofacitinib', 'Xeljanz',
#     'Filgotinib', 'Jyseleca', 'Deucravacitinib', 'Sotyktu',
#     'Delgocitinib', 'Corectim', 'Ruxolitinib', 'Jakavi',
#     'Opzelura', 'Peficitinib', 'Smyraf'
# ]

# # Initialize graph
# G_tiktok = nx.Graph()

# # Helper to add or update node
# def add_node(graph, node, **attrs):
#     if graph.has_node(node):
#         graph.nodes[node].update(attrs)
#     else:
#         graph.add_node(node, **attrs)

# # Build graph
# for _, row in tiktok_df.iterrows():
#     post_id = str(row['post_id']) if pd.notnull(row['post_id']) else None
#     author_id = row['author_unique_id'] if pd.notnull(row['author_unique_id']) else "Unknown_User"
#     desc = str(row.get('translated_desc', ''))
#     vid_text = str(row.get('translated_video_text', ''))
#     text = str(row.get('translated_text', ''))
#     combined_text = ' '.join([desc, vid_text, text]).strip()

#     if not post_id:
#         continue

#     # Add nodes
#     add_node(G_tiktok, post_id, type='video', content=combined_text)
#     add_node(G_tiktok, author_id, type='user')
#     G_tiktok.add_edge(author_id, post_id, type='authorship')

#     for drug in drugs:
#         if drug.lower() in combined_text.lower():
#             add_node(G_tiktok, drug, type='drug')
#             G_tiktok.add_edge(post_id, drug, type='mentions')

# # --- Community Detection ---

# # Louvain Algorithm
# louvain_partition = community_louvain.best_partition(G_tiktok)

# # Assign community as node attribute
# nx.set_node_attributes(G_tiktok, louvain_partition, 'louvain_community')

# # Number of communities detected
# num_louvain_communities = len(set(louvain_partition.values()))

# # --- Girvan-Newman Algorithm (Top 5 communities only) ---
# girvan_communities = list(islice(girvan_newman(G_tiktok), 5))
# gn_top_partition = {node: i for i, community in enumerate(girvan_communities[-1]) for node in community}

# # Assign community as node attribute
# nx.set_node_attributes(G_tiktok, gn_top_partition, 'girvan_newman_community')

# # Number of communities detected by Girvan-Newman
# num_gn_communities = len(set(gn_top_partition.values()))

# # Return counts
# print(num_louvain_communities)
# print (num_gn_communities)


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # pip install python-louvain
from networkx.algorithms.community import girvan_newman
from itertools import islice

# Load TikTok data
tiktok_df = pd.read_csv("tiktok_data.csv")

# Define drug list
drugs = ['Upadacitinib', 'Rinvoq', 'Jak inhibitors', 'Abrocitinib', 'Cibinqo', 'Baricitinib', 'Olumiant',
         'Ritlecitinib', 'Litfulo', 'Tofacitinib', 'Xeljanz', 'Filgotinib', 'Jyseleca', 'Deucravacitinib',
         'Sotyktu', 'Delgocitinib', 'Corectim', 'Ruxolitinib', 'Jakavi', 'Opzelura', 'Peficitinib', 'Smyraf']

# Create Graph
G = nx.Graph()

def add_node(G, node, **attrs):
    if G.has_node(node):
        G.nodes[node].update(attrs)
    else:
        G.add_node(node, **attrs)

for _, row in tiktok_df.iterrows():
    post_id = str(row['post_id']) if pd.notnull(row['post_id']) else None
    author_id = row['author_unique_id'] if pd.notnull(row['author_unique_id']) else "Unknown_User"
    combined_text = ' '.join([str(row.get(f, '')) for f in ['translated_desc', 'translated_video_text', 'translated_text']])

    if not post_id:
        continue

    add_node(G, post_id, type='video', content=combined_text)
    add_node(G, author_id, type='user')
    G.add_edge(author_id, post_id, type='authorship')

    for drug in drugs:
        if drug.lower() in combined_text.lower():
            add_node(G, drug, type='drug')
            G.add_edge(post_id, drug, type='mentions')

# --- Louvain Community Detection ---
louvain_partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, louvain_partition, 'louvain_community')
print(f"Louvain detected {len(set(louvain_partition.values()))} communities.")

# --- Girvan-Newman (top 5 only) ---
gn_communities = list(islice(girvan_newman(G), 5))
gn_top_partition = {node: i for i, comm in enumerate(gn_communities[-1]) for node in comm}
nx.set_node_attributes(G, gn_top_partition, 'girvan_newman_community')
print(f"Girvan-Newman detected {len(set(gn_top_partition.values()))} communities.")

# Optional: Save node communities to CSV
nodes_data = []
for node, data in G.nodes(data=True):
    nodes_data.append({
        'Node': node,
        'Type': data.get('type'),
        'Louvain Community': data.get('louvain_community'),
        'Girvan-Newman Community': data.get('girvan_newman_community')
    })

pd.DataFrame(nodes_data).to_csv("tiktok_community_nodes.csv", index=False)
print("Saved community results to 'tiktok_community_nodes.csv'")
