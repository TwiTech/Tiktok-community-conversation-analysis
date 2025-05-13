import pandas as pd
import networkx as nx

# Load the TikTok data
tiktok_path = "tiktok_data.csv"
tiktok_df = pd.read_csv(tiktok_path)

# Standard drug list
drugs = [
    'Upadacitinib', 'Rinvoq', 'Jak inhibitors',
    'Abrocitinib', 'Cibinqo',
    'Baricitinib', 'Olumiant',
    'Ritlecitinib', 'Litfulo',
    'Tofacitinib', 'Xeljanz',
    'Filgotinib', 'Jyseleca',
    'Deucravacitinib', 'Sotyktu',
    'Delgocitinib', 'Corectim',
    'Ruxolitinib', 'Jakavi', 'Opzelura',
    'Peficitinib', 'Smyraf'
]

# Initialize a directed graph
G_tiktok = nx.DiGraph()

# Function to add a node with updates
def add_node(graph, node, **attrs):
    if graph.has_node(node):
        graph.nodes[node].update(attrs)
    else:
        graph.add_node(node, **attrs)

# Construct Graph from TikTok Data
for _, row in tiktok_df.iterrows():
    post_id = str(row['post_id']) if pd.notnull(row['post_id']) else None
    author_id = row['author_unique_id'] if pd.notnull(row['author_unique_id']) else "Unknown_User"
    desc = str(row.get('translated_desc', ''))
    vid_text = str(row.get('translated_video_text', ''))
    text = str(row.get('translated_text', ''))
    combined_text = ' '.join([desc, vid_text, text]).strip()

    if not post_id:
        continue

    # Add video (post) node
    add_node(
        G_tiktok,
        post_id,
        type='video',
        content=combined_text,
        timestamp=row.get('createTime'),
        author=author_id,
        digg_count=row.get('digg_count'),
        comment_count=row.get('reply_comment_total'),
        url=row.get('post_url')
    )

    # Add user (author) node
    add_node(G_tiktok, author_id, type='user', username=author_id)
    G_tiktok.add_edge(author_id, post_id, type='user_to_video')

    # Add drug nodes and connect
    for drug in drugs:
        if drug.lower() in combined_text.lower():
            add_node(G_tiktok, drug, type='drug')
            G_tiktok.add_edge(post_id, drug, type='video_to_drug')

# Summary
node_count = G_tiktok.number_of_nodes()
edge_count = G_tiktok.number_of_edges()

print (node_count)  
print (edge_count)
