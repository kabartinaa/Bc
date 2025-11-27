import os
import random
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np

# --- 1. Configuration (Updated for GNN Project) ---
ciphers = ["AES", "DES", "3DES", "Blowfish", "Twofish", "RC4", "CAST", "IDEA"]
modes = ["ECB", "CBC"]
samples_per_class = 200
dataset_dir = "dataset"
NUM_CLASSES = len(ciphers) * len(modes) # Should be 16

# Placeholder for the 256D LSTM feature vector
LSTM_FEATURE_DIM = 256 
INPUT_FEATURE_DIM = LSTM_FEATURE_DIM + 4 # 260D total features (256 + 4 topology)

# Opcode vocabulary for simulation
opcode_vocab = ["MOV", "ADD", "SUB", "XOR", "AND", "OR", "CMP", "JMP", "CALL", "RET"]

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# -------------------------
# 2. Helper Functions
# -------------------------

def generate_cfg_graph(variant_id):
    """
    Generate a random CFG graph structure and assign initial features.
    NOTE: The graph structure and features are randomized. In a real project, 
    they must be extracted from compiled binaries.
    """
    num_nodes = random.randint(15, 30) # Increased node count for complexity
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)

    # Random edges (ensures some graph variety)
    for i in range(num_nodes):
        # Forward edges
        if i < num_nodes - 1:
            G.add_edge(i, i + 1)
        # Random backward/branch edges
        if random.random() < 0.2 and i > 0:
            G.add_edge(i, random.randint(0, i - 1))

    # --- SIMULATION OF LSTM FEATURES (256D) ---
    # In a real pipeline, this would be lstm_feature_model.predict(opcode_sequences)
    
    node_features = []
    opcode_sequences = [] # Needed for the real LSTM model input

    for i in G.nodes():
        # Generate random opcode sequence for 'real' data
        node_opcodes = random.choices(opcode_vocab, k=random.randint(5, 12))
        opcode_sequences.append(node_opcodes)
        
        # 1. Base Feature (Simulated LSTM output) - 256D
        # Using uniform random data ensures the feature vector has a distinct size, 
        # forcing the GNN to learn to differentiate based on the topology features we add.
        lstm_sim_vector = np.random.randn(LSTM_FEATURE_DIM).astype(np.float32)
        node_features.append(lstm_sim_vector)
    
    # --- END SIMULATION ---
    
    # 2. Calculate Advanced Topology Features (4D)
    
    # **PageRank (1D):** Global influence of the node
    # Note: NetworkX pagerank returns a dictionary; convert to list based on node order
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    pagerank_list = np.array([pagerank_dict[n] for n in G.nodes()])

    # **Clustering Coefficient (1D):** Local community structure
    # For a directed graph, we use the average clustering or local clustering as a measure
    clustering_dict = nx.clustering(G)
    clustering_list = np.array([clustering_dict[n] for n in G.nodes()])

    # **In/Out Degree (2D):** Local connectivity
    in_degree_list = np.array([G.in_degree(n) for n in G.nodes()])
    out_degree_list = np.array([G.out_degree(n) for n in G.nodes()])

    # --- 3. Concatenate and Normalize Features (260D) ---
    
    # Reshape all 1D arrays to (N_nodes, 1) for concatenation
    pagerank = pagerank_list.reshape(-1, 1)
    clustering = clustering_list.reshape(-1, 1)
    in_degree = in_degree_list.reshape(-1, 1)
    out_degree = out_degree_list.reshape(-1, 1)

    # Concatenate: (N x 256) + (N x 4) = (N x 260)
    lstm_tensor = torch.tensor(np.array(node_features), dtype=torch.float)
    topo_tensor = torch.tensor(np.concatenate([in_degree, out_degree, clustering, pagerank], axis=1), dtype=torch.float)
    
    # Final feature matrix: 260D
    x = torch.cat([lstm_tensor, topo_tensor], dim=1)
    
    # IMPORTANT: Normalize the topological features (columns 256:260) to prevent 
    # them from dominating the LSTM features, as GNNs are sensitive to feature magnitude.
    # Note: A full pipeline would normalize all features based on global min/max across the entire dataset.
    
    
    # Edge index remains the same
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    
    return G, x, edge_index, opcode_sequences


# -------------------------
# 3. Generate Dataset Loop
# -------------------------
class_idx = 0
all_graph_features = []
all_labels = []

for cipher in ciphers:
    for mode in modes:
        class_name = f"{cipher}-{mode}"
        folder = os.path.join(dataset_dir, class_name)
        os.makedirs(folder, exist_ok=True)
        
        print(f"Generating class {class_idx+1}/{NUM_CLASSES}: {class_name} with {samples_per_class} samples...")

        for variant_id in range(1, samples_per_class + 1):
            # Generate CFG + 260D node features + opcode sequence
            G, x, edge_index, opcode_seq = generate_cfg_graph(variant_id)
            
            # PyG data object
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([class_idx]))
            
            # --- Optional: Collect graph-level features for a simple baseline (XGBoost) ---
            # NOTE: The original graph_features_for_ml function is removed for simplicity 
            # and focus on the GNN data object 'data'. You can re-add it if needed.
            
            # Save data
            torch.save({
                'pyg_data': data,
                'opcode_seq': opcode_seq,
                'label': class_idx
            }, os.path.join(folder, f"data_{variant_id:03d}.pt"))
            
        class_idx += 1

print("\n--- Dataset Generation Complete ---")
print(f"Successfully generated 16 classes x {samples_per_class} samples = {NUM_CLASSES * samples_per_class} total synthetic samples.")
print(f"Each graph now has {INPUT_FEATURE_DIM}D node features, combining simulated LSTM and 4 key topology features.")
