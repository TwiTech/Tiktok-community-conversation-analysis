import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load TikTok data and community assignments
tiktok_path = "tiktok_data.csv"
community_path = "tiktok_community_nodes.csv"

df_tiktok = pd.read_csv(tiktok_path)
df_community = pd.read_csv(community_path)

# Define drug list
drugs = ['Upadacitinib', 'Rinvoq', 'Jak inhibitors', 'Abrocitinib', 'Cibinqo', 'Baricitinib', 'Olumiant',
         'Ritlecitinib', 'Litfulo', 'Tofacitinib', 'Xeljanz', 'Filgotinib', 'Jyseleca', 'Deucravacitinib',
         'Sotyktu', 'Delgocitinib', 'Corectim', 'Ruxolitinib', 'Jakavi', 'Opzelura', 'Peficitinib', 'Smyraf']

# Step 1: Rebuild the full interaction graph
G = nx.Graph()

for _, row in df_tiktok.iterrows():
    post_id = str(row['post_id']) if pd.notnull(row['post_id']) else None
    user_id = row['author_unique_id'] if pd.notnull(row['author_unique_id']) else "Unknown_User"
    combined_text = ' '.join([str(row.get(f, '')) for f in ['translated_desc', 'translated_video_text', 'translated_text']])

    if not post_id or not combined_text.strip():
        continue

    G.add_node(post_id, type='video', content=combined_text)
    G.add_node(user_id, type='user')
    G.add_edge(user_id, post_id, type='user_to_video')

    for drug in drugs:
        if drug.lower() in combined_text.lower():
            G.add_node(drug, type='drug')
            G.add_edge(post_id, drug, type='video_to_drug')

# Step 2: Helper function for plotting communities
def plot_community(community_ids, community_type='Louvain Community', title_prefix='Louvain'):
    community_nodes_map = defaultdict(list)
    for _, row in df_community.iterrows():
        if row[community_type] in community_ids:
            community_nodes_map[row[community_type]].append(row['Node'])

    for community_id in community_ids:
        nodes_in_community = community_nodes_map[community_id]

        # Add connected drug nodes
        drug_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'drug']
        connected_drugs = [drug for node in nodes_in_community for drug in drug_nodes if G.has_edge(node, drug)]
        subgraph_nodes = set(nodes_in_community + connected_drugs)

        G_sub = G.subgraph(subgraph_nodes).copy()

        # Color and label nodes
        node_colors = []
        node_labels = {}
        for n in G_sub.nodes():
            n_type = G_sub.nodes[n].get('type')
            if n_type == 'user':
                node_colors.append('skyblue')
            elif n_type == 'video':
                node_colors.append('orange')
            elif n_type == 'drug':
                node_colors.append('red')
                node_labels[n] = n
            else:
                node_colors.append('gray')

        plt.figure(figsize=(13, 10))
        pos = nx.spring_layout(G_sub, seed=42, k=0.3)
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=60, alpha=0.8)
        nx.draw_networkx_edges(G_sub, pos, alpha=0.5)
        nx.draw_networkx_labels(G_sub, pos, labels=node_labels, font_size=9, font_color='black')
        plt.title(f"{title_prefix} Community {community_id}: Users, Videos, and Drug Mentions")
        plt.axis('off')
        plt.show()

# Step 3: Plot top 3 Louvain communities
top3_louvain_ids = df_community[df_community['Type'] == 'video']['Louvain Community'].value_counts().head(3).index.tolist()
plot_community(top3_louvain_ids, community_type='Louvain Community', title_prefix='Louvain')

# Step 4: Plot top 3 Girvan–Newman communities
top3_gn_ids = df_community[df_community['Type'] == 'video']['Girvan-Newman Community'].value_counts().head(3).index.tolist()
plot_community(top3_gn_ids, community_type='Girvan-Newman Community', title_prefix='Girvan–Newman')
