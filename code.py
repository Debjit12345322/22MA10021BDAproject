# STEP 1: Extract 100K eligible nodes
import pandas as pd
import networkx as nx
import pickle
import os
import time

# CONFIG
chunk_size = 1000
sleep_minutes = 0
dataset_file = "/kaggle/input/datafinal/dataset_final.csv"
save_dir = "eligible_chunks"
final_output_file = "final_node_ids.pkl"
target_node_count = 100000

# Ensure output folder exists
os.makedirs(save_dir, exist_ok=True)

# Load dataset and graph
print("Loading dataset...")
df = pd.read_csv(dataset_file)
G = nx.from_pandas_edgelist(df, source="Follower", target="Target", create_using=nx.DiGraph())
all_nodes = list(G.nodes())
total_chunks = (len(all_nodes) + chunk_size - 1) // chunk_size

print(f"Total nodes in graph: {len(all_nodes)}")
print(f"Target: {target_node_count} eligible profiles")

# Track final list
final_eligible = []

# Process in chunks
for chunk_id in range(total_chunks):
    if len(final_eligible) >= target_node_count:
        print("Target reached.")
        break

    filename = os.path.join(save_dir, f"eligible_nodes_chunk_{chunk_id}.pkl")
    if os.path.exists(filename):
        print(f"Chunk {chunk_id} already processed. Loading from disk.")
        with open(filename, "rb") as f:
            eligible = pickle.load(f)
        final_eligible.extend(eligible)
        continue

    print(f"\n[Chunk {chunk_id}] Processing nodes {chunk_id * chunk_size} to {(chunk_id + 1) * chunk_size}...")
    nodes_chunk = all_nodes[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
    eligible = []

    for i, node in enumerate(nodes_chunk):
        ego = nx.ego_graph(G, node, radius=3)
        if len(ego) >= 1000:
            eligible.append(node)
        if i % 100 == 0:
            print(f"  Checked {i} nodes... Eligible this chunk: {len(eligible)}")

    with open(filename, "wb") as f:
        pickle.dump(eligible, f)
    print(f"[Chunk {chunk_id}] Saved {len(eligible)} nodes to {filename}")

    final_eligible.extend(eligible)

    print(f"[Chunk {chunk_id}] Sleeping for {sleep_minutes} minutes...")
    time.sleep(sleep_minutes * 60)

# Final 100K trimming and save
final_eligible = final_eligible[:target_node_count]
with open(final_output_file, "wb") as f:
    pickle.dump(final_eligible, f)
print(f"\n✅ Saved {len(final_eligible)} eligible node IDs to {final_output_file}")


# STEP 2: Generate Node2Vec embeddings for each ego network
from node2vec import Node2Vec
from tqdm import tqdm

embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

# Load final 100K nodes
with open(final_output_file, "rb") as f:
    eligible_nodes = pickle.load(f)

# Generate embeddings
for idx, node in enumerate(tqdm(eligible_nodes)):
    out_path = os.path.join(embedding_dir, f"embedding_{idx}.pkl")
    if os.path.exists(out_path):
        continue

    ego_net = nx.ego_graph(G, node, radius=3)
    if ego_net.number_of_nodes() < 1000:
        continue

    try:
        n2v = Node2Vec(ego_net, dimensions=64, walk_length=30, num_walks=200, workers=1, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        # Save embedding vectors (node -> vector)
        embeddings = {n: model.wv[n] for n in ego_net.nodes() if n in model.wv}
        with open(out_path, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        print(f"Failed for node {node}: {e}")


# STEP 3: Train a model using embeddings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# For demonstration, we will classify ego network centers based on average embedding vector
X = []
y = []

# Load embeddings and create dummy labels (e.g., binary label based on degree)
for idx, node in enumerate(tqdm(eligible_nodes)):
    emb_path = os.path.join(embedding_dir, f"embedding_{idx}.pkl")
    if not os.path.exists(emb_path):
        continue

    with open(emb_path, "rb") as f:
        emb_dict = pickle.load(f)

    if node not in emb_dict:
        continue

    # Feature: average of all vectors
    vectors = np.array(list(emb_dict.values()))
    avg_vector = np.mean(vectors, axis=0)
    X.append(avg_vector)

    # Dummy label: high vs low degree
    degree = G.degree(node)
    y.append(1 if degree > 20 else 0)

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Optional: PCA to reduce dimensions
pca = PCA(n_components=32)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy on test set: {acc:.4f}")
