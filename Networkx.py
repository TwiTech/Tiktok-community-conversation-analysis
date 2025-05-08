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

# Convert NetworkX graph to DataFrames for export

# # Extract nodes
# nodes_data = []
# for node, attrs in G.nodes(data=True):
#     attrs["Node"] = node  # Ensure the ID is included
#     nodes_data.append(attrs)

# nodes_df = pd.DataFrame(nodes_data)

# # Extract edges
# edges_data = []
# for source, target, attrs in G.edges(data=True):
#     edges_data.append({"Source": source, "Target": target, "type": attrs["type"]})

# edges_df = pd.DataFrame(edges_data)

# # Save to CSV for Neo4j import
# nodes_csv_path = "nodes.csv"
# edges_csv_path = "edges.csv"

# nodes_df.to_csv(nodes_csv_path, index=False)
# edges_df.to_csv(edges_csv_path, index=False)

# print(f"Nodes exported to: {nodes_csv_path}")
# print(f"Edges exported to: {edges_csv_path}")

import random
import matplotlib.pyplot as plt
import networkx as nx

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5, 
                  pos=None, parent=None):
    """
    Recursively assign positions to nodes in a hierarchy (tree) layout.
    
    G:        The graph (assumed to be a tree or tree-like).
    root:     The root node of current branch.
    width:    Horizontal space allotted for this branch.
    vert_gap: Gap between levels of hierarchy.
    xcenter:  Horizontal center of this branch.
    pos:      A dict storing positions of nodes.
    parent:   Parent of this branch's root.
    
    Returns a dict: node -> (x, y)
    """
    if pos is None:
        pos = {root: (xcenter, 0)}
    else:
        pos[root] = (xcenter, pos[parent][1] - vert_gap) if parent else (xcenter, 0)

    children = list(G.successors(root))  # or G.neighbors(root) if undirected
    if not children:
        return pos

    # If there are children, divide width equally among them
    dx = width / max(1, len(children))
    nextx = xcenter - width/2 - dx/2

    for child in children:
        nextx += dx
        pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                            xcenter=nextx, pos=pos, parent=root)
    return pos

def build_reddit_hierarchy_subgraph(G, drug_node, max_depth=2):
    """
    Create a BFS "hierarchy" from 'drug_node' up to 'max_depth' levels,
    excluding user nodes. Assumes edges are oriented from 'drug' -> 'post' -> 'comment'
    or similar. If your edges are reversed, you might need to tweak directions.
    """
    # Only keep nodes of type in {drug, post, comment}
    valid_nodes = set(
        n for n, d in G.nodes(data=True)
        if d.get("type") in ("drug", "post", "comment")
    )
    
    if drug_node not in valid_nodes:
        print(f"Drug node '{drug_node}' not in the filtered set or not a 'drug' node.")
        return None

    # We'll do a BFS up to 'max_depth' edges away from drug_node
    # but treat the graph as directed from drug -> post -> comment.
    visited = set([drug_node])
    level = {drug_node: 0}
    queue = [drug_node]
    
    # Build a small "directed" subgraph
    H = nx.DiGraph()
    H.add_node(drug_node, **G.nodes[drug_node])
    
    while queue:
        current = queue.pop(0)
        current_depth = level[current]
        if current_depth >= max_depth:
            continue
        
        # For each neighbor of 'current' in G
        # we only follow edges that keep the "drug->post->comment" hierarchy
        # i.e. if current is 'drug', we follow edges to posts or comments,
        # if current is 'post', we follow edges to comments, etc.
        for neighbor in G.successors(current):
            # Exclude user nodes
            if neighbor not in valid_nodes:
                continue

            # If not visited
            if neighbor not in visited:
                visited.add(neighbor)
                level[neighbor] = current_depth + 1
                # Add neighbor to subgraph with attributes
                H.add_node(neighbor, **G.nodes[neighbor])
                queue.append(neighbor)
            
            # Add the edge to H
            if not H.has_edge(current, neighbor):
                H.add_edge(current, neighbor, **G[current][neighbor])
    
    return H

# ------------------
# Usage Example
# ------------------

# Suppose we want to pick a random drug node that actually exists in the graph
drug_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "drug"]
if not drug_nodes:
    print("No drug nodes found in the graph!")
else:
    chosen_drug = random.choice(drug_nodes)
    print("Chosen Drug Node:", chosen_drug)
    
    # Build a BFS subgraph up to 2 levels away from that drug node
    subH = build_reddit_hierarchy_subgraph(G, chosen_drug, max_depth=2)
    
    if subH and subH.number_of_nodes() > 1:
        # Visualize it in a hierarchy layout
        # We'll treat 'chosen_drug' as the root
        pos = hierarchy_pos(subH, root=chosen_drug)  # or neighbors(...) if undirected

        plt.figure(figsize=(10, 8))
        nx.draw_networkx(
            subH, pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            node_size=800,
            font_size=8
        )
        plt.title(f"Hierarchy from Drug Node '{chosen_drug}' (max_depth=2)")
        plt.axis('off')
        plt.show()
    else:
        print("No connected subgraph found from that drug node (or subgraph is empty).")
