import pandas as pd
import networkx as nx

# Load the datasets
posts_path = "reddit_posts.csv"  # Update path if necessary
comments_path = "reddit_comments.csv"

posts_df = pd.read_csv(posts_path, low_memory=False)
comments_df = pd.read_csv(comments_path, low_memory=False)

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
G = nx.DiGraph()

# Function to add a node with updates
def add_node(graph, node, **attrs):
    if graph.has_node(node):
        graph.nodes[node].update(attrs)  # Update existing attributes
    else:
        graph.add_node(node, **attrs)  # Create new node

# Construct Graph from Posts
for _, row in posts_df.iterrows():
    post_id = str(row['Post ID'])
    post_title = row['Post Title']
    post_body = row['Post Body'] if pd.notnull(row['Post Body']) else row['Post Title']
    
    add_node(
        G,
        post_id,
        type='post',
        title=post_title,
        post_body=post_body,
        timestamp=row['Post Timestamp'],
        author=row['Post Author'],
        score=row['Score'],
        subreddit=row['Subreddit'],
        url=row['URL']
    )

    # Ensure author is treated as a user node
    user = row['Post Author']
    if pd.isnull(user) or user.strip() == "":
        user = "Unknown_User"

    add_node(G, user, type='user', username=user, author=user)
    G.add_edge(user, post_id, type='user_to_post')

    # Detect and add drug mentions in post content
    for drug in drugs:
        if isinstance(post_title, str) and drug.lower() in post_title.lower():
            add_node(G, drug, type='drug')
            G.add_edge(post_id, drug, type='post_to_drug')

        if isinstance(post_body, str) and drug.lower() in post_body.lower():
            add_node(G, drug, type='drug')
            G.add_edge(post_id, drug, type='post_to_drug')

# Construct Graph from Comments
for _, row in comments_df.iterrows():
    comment_id = str(row['Comment ID'])
    parent_id = str(row['Parent ID']) if pd.notnull(row['Parent ID']) else None
    comment_body = row['Comment Body']
    comment_author = row['Comment Author']

    # Ensure author is treated as a user node
    if pd.isnull(comment_author) or comment_author.strip() == "":
        comment_author = "Unknown_User"

    # Add comment node
    add_node(
        G,
        comment_id,
        type='comment',
        body=comment_body,
        timestamp=row['Comment Timestamp'],
        author=comment_author,
        score=row['Comment Score'],
        parent_id=parent_id,
        link_to_post=row['Link to Post']
    )

    # Ensure comment author exists as a user node
    add_node(G, comment_author, type='user', username=comment_author, author=comment_author)
    G.add_edge(comment_author, comment_id, type='user_to_comment')

    # Connect comment to its parent (post or another comment)
    if parent_id and G.has_node(parent_id):
        parent_type = G.nodes[parent_id]['type']
        edge_type = 'comment_to_post' if parent_type == 'post' else 'comment_to_comment'
        G.add_edge(comment_id, parent_id, type=edge_type)

    # Detect and add drug mentions in comment body
    if isinstance(comment_body, str):
        for drug in drugs:
            if drug.lower() in comment_body.lower():
                add_node(G, drug, type='drug')
                G.add_edge(comment_id, drug, type='comment_to_drug')

# Print total node and edge counts
print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())

