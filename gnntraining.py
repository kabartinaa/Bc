import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset , DataLoader
#from torch_geometric.loader import 
from torch_geometric.nn import GCNConv, global_mean_pool , GATConv
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- PATHS AND CONSTANTS ---
BASE_DIR = os.path.expanduser("/home/kali/Documents/Block_cipher")
SPLIT_TRAIN_DATA = os.path.join(BASE_DIR, "Split_data")
FEATURE_SAVE_PATH = os.path.join(SPLIT_TRAIN_DATA, "gnn_features")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models")

GNN_SPLIT_FILE = os.path.join(SPLIT_TRAIN_DATA, "gnn_train_val.pkl")
X_SEQ_FILE = os.path.join(FEATURE_SAVE_PATH, "X_seq_features_128D.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5')

# --- HYPER PARAMETERS ---
CHANNELS = 32
EPOCHS =200
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64 # PyG DataLoader handles this size efficiently

# NOTE: These are defined by the LSTM model
LSTM_FEATURE_DIM = 256
MAX_SEQUENCE_LENGTH = 74


INPUT_FEATURE_DIM = LSTM_FEATURE_DIM


import shutil
import os

root_dir = os.path.join(BASE_DIR, "pyg_dataset_cache") 
os.makedirs(root_dir, exist_ok=True)


#processed_train_dir = os.path.join(root_dir, "train", "processed")
#processed_val_dir = os.path.join(root_dir, "val", "processed")

#shutil.rmtree(processed_train_dir, ignore_errors=True)
#shutil.rmtree(processed_val_dir, ignore_errors=True)



# --- 1. LSTM Feature Extractor Setup (As provided in your snippet) ---

try:
	lstm_feature_extractor = load_model(LSTM_MODEL_PATH, compile=False)
	lstm_feature_extractor.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

	lstm_output_layer = lstm_feature_extractor.layers[1].output
	lstm_feature_model = Model(inputs=lstm_feature_extractor.inputs, outputs=lstm_output_layer)
	#print(f"lstm_feature_model : {lstm_feature_model} \n")
	LSTM_FEATURE_DIM = lstm_output_layer.shape[1]
	#print(f"LSTM_FEATURE_DIM : {LSTM_FEATURE_DIM} \n")
	print("SUCCESS: LSTM feature model loaded and configured for feature extraction.\n")
	
except Exception as e:
	print(f"Error on loading LSTM model {e}")
	
	
# - - - - convert a NetworkX graph into a PyTorch Geometric (PyG) Data object - - - -
def convert_nx_to_pyg_data(nx_graph , label_id , lstm_feature_model):
	nx_graph_relabel = nx.convert_node_labels_to_integers(nx_graph, first_label=0)
	#print(f"nx_graph_relabel : {nx_graph_relabel}")
	node_features_list = [
        nx_graph_relabel.nodes[node_id].get('sequence', [1] * MAX_SEQUENCE_LENGTH) for node_id in nx_graph_relabel.nodes ]
	
	
	padded_sequences = pad_sequences(
        node_features_list, 
        maxlen=MAX_SEQUENCE_LENGTH, 
        padding='post', 
        truncating='post' )
        
	if hasattr(lstm_feature_model, 'predict'):
		lstm_features = lstm_feature_model.predict(padded_sequences)
	elif callable(lstm_feature_model):
		lstm_features = lstm_feature_model(padded_sequences)
	else:
		lstm_features = np.ones((len(padded_sequences), LSTM_FEATURE_DIM), dtype=np.float32)
		
	print(lstm_features)
	
	# 3. Create Node Feature Tensor (X)
	
	
	
	norms = np.linalg.norm(lstm_features, axis=1, keepdims=True)
# Avoid division by zero for any zero vectors
	norms[norms == 0] = 1.0 
	normalized_features = lstm_features / norms

# 3. Create Node Feature Tensor (X) - Use the normalized features
	x = torch.tensor(normalized_features, dtype=torch.float)
	
	print(f"content inside convert_nx_to_pyg_data \n\n")
	
	print("Feature Mean:", x.mean().item(), "Feature Std:", x.std().item())

	
	
	
	
	
    
    # 4. Edge Index: Convert NetworkX edges to PyG's COO format
	edge_list = list(nx_graph_relabel.edges())
	if not edge_list:
		edge_index = torch.empty((2, 0), dtype=torch.long)
	else:
        # Standard conversion for PyG
		edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 5. Label
	y = torch.tensor([label_id], dtype=torch.long)
	
	print("Edges:", edge_index.shape[1])

    
	return Data(x=x, edge_index=edge_index, y=y)
	
#  - - - - - - convert_nx_to_pyg_data() converts one graph at a time - -- -- - - 

# - - - - - - - BlockCipherDataset organizes, stores, and loads all graph samples - - - - - - 

class BlockCipherDataset(Dataset):
	def __init__(self, root, A_list,  Y_one_hot, label_to_id, lstm_feature_model, transform=None, pre_transform=None):
		self.A_list = A_list
		self.Y_one_hot = Y_one_hot
		self.label_to_id = label_to_id
		self.lstm_feature_model = lstm_feature_model
		super().__init__(root, transform, pre_transform)
		
		
	@property
	def raw_file_names(self):
		return []
		
	@property
	def processed_file_names(self):
		return [f'data_{i}.pt' for i in range(len(self.A_list))]

	def download(self):
		pass

	def process(self):
		shutil.rmtree(self.processed_dir, ignore_errors=True)
		os.makedirs(self.processed_dir, exist_ok=True)
		for i, (graph, label) in enumerate(zip(self.A_list, self.Y_one_hot)):
			label_id = np.argmax(label) # Get the index ID from one-hot
			pyg_data = convert_nx_to_pyg_data(graph, label_id, self.lstm_feature_model)
			torch.save(pyg_data, os.path.join(self.processed_dir, f'data_{i}.pt'))

	def len(self):
		return len(self.A_list)

	def get(self, idx):
		file_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        # CRITICAL FIX: Add weights_only=False to bypass the security check 
        # for loading data objects instead of model weights.
		return torch.load(file_path, weights_only=False)



try:
	with open(GNN_SPLIT_FILE, 'rb') as f:
		gnn_data_splits = pickle.load(f)

	A_train = gnn_data_splits['X_train_gnn']
#print(A_train)
	y_train_raw = gnn_data_splits['y_train']
	A_val = gnn_data_splits['X_val_gnn']
	y_val_raw = gnn_data_splits['y_val']

	all_unique_labels = sorted(list(set(y_train_raw.tolist() + y_val_raw.tolist())))

	NUM_CLASSES = len(all_unique_labels)
	#NUM_CLASSES = 16
	print(f"num of classes : {NUM_CLASSES}")
	label_to_id = {label:i for i, label in enumerate(all_unique_labels)}
	
	
	y_train_encoded = np.array([label_to_id[label] for label in y_train_raw])
	y_val_encoded = np.array([label_to_id[label] for label in y_val_raw])
    
	y_train_one_hot = F.one_hot(torch.tensor(y_train_encoded), num_classes=NUM_CLASSES).numpy()
	#print(f"y_train_one_hot :  {y_train_one_hot}")
	y_val_one_hot = F.one_hot(torch.tensor(y_val_encoded), num_classes=NUM_CLASSES).numpy()
	
except Exception as e:
	print(f"Error in value encoding {e}")
	
	
# Create Dataset and DataLoader
# NOTE: Using a simple dummy root directory for demonstration

train_dataset = BlockCipherDataset(
    root=os.path.join(root_dir, "train"),
    A_list=A_train,
    Y_one_hot=y_train_one_hot,
    label_to_id=label_to_id,
    lstm_feature_model=lstm_feature_model
)

val_dataset = BlockCipherDataset(
    root=os.path.join(root_dir, "val"),
    A_list=A_val,
    Y_one_hot=y_val_one_hot,
    label_to_id=label_to_id,
    lstm_feature_model=lstm_feature_model
)



import torch
import numpy as np

# y_train_encoded should be your encoded labels for training samples
class_sample_count = np.array([np.sum(y_train_encoded == t) for t in range(NUM_CLASSES)])
weight_per_class = 1.0 / class_sample_count
samples_weight = np.array([weight_per_class[t] for t in y_train_encoded])

samples_weight_tensor = torch.DoubleTensor(samples_weight)


from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(samples_weight_tensor, len(samples_weight_tensor), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"\nTraining with {len(train_dataset)} graphs, {len(train_loader)} batches.")


# --- 4. PYTORCH GNN MODEL DEFINITION ---

class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate , heads=4):
		super(GNN, self).__init__()
		torch.manual_seed(12345)
		# LAYER 1
		self.conv1 = GATConv(in_channels, hidden_channels , heads=heads, dropout=dropout_rate)
		#Layer 2 — GCNConv
		#self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
		self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout_rate)
		#self.lin = torch.nn.Linear(hidden_channels, out_channels) # Changed input from 16 to 32
		self.lin = torch.nn.Linear(hidden_channels, out_channels)
		self.dropout_rate = dropout_rate
		
	def forward(self , data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		# Layer 1
		x = self.conv1(x, edge_index)
		x = F.elu(x)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Layer 2
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		#x = F.dropout(x, p=self.dropout_rate, training=self.training)
		
		
	# pooling job : For each graph: take the mean of all node feature rows that belong to that graph.
		x = global_mean_pool(x, batch)
		
	# linear for prediction
		x = self.lin(x)
		
	# softmax converts prediction too probability
		return x # Use log_softmax for NLLLoss

#Final Intuition (Very Simple)
#GCNConv

#👉 "Extract features from neighboring nodes"

#ReLU

#👉 "Keep only positive, meaningful activations"

#Dropout

#👉 "Prevent overfitting and improve generalization"

#And repeating it twice → a deeper, more powerful GNN.

# same procedure followed for 2 layers

model = GNN(
    in_channels=INPUT_FEATURE_DIM, 
    hidden_channels=CHANNELS, 
    out_channels=NUM_CLASSES, 
    dropout_rate=DROPOUT_RATE
)

#optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE) # Change from Adam
# Change the optimizer initialization:
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.NLLLoss()










print(f"\nPyTorch GNN Model:\n{model}")


# --- 5. TRAINING AND VALIDATION FUNCTIONS ---

def train():
	model.train()
	total_loss = 0
	correct = 0
	for data in train_loader:
		optimizer.zero_grad()
		out = model(data)
		loss = criterion(out, data.y) 
		loss.backward()
		optimizer.step()
        
		total_loss += loss.item() * data.num_graphs
		pred = out.argmax(dim=1)
		correct += (pred == data.y).sum().item()
		
		print("Train preds:", pred[:20])
		print("Train labels:", data.y[:20])

        
	return total_loss / len(train_dataset), correct / len(train_dataset)
	
@torch.no_grad()
def test(loader):
	model.eval()
	total_loss = 0
	correct = 0
    
	for data in loader:
		out = model(data)
		loss = criterion(out, data.y)
		total_loss += loss.item() * data.num_graphs
		pred = out.argmax(dim=1)
		correct += (pred == data.y).sum().item()        
	return total_loss / len(loader.dataset), correct / len(loader.dataset)
	
	
	
# --- 6. EXECUTE TRAINING LOOP ---

print("\nStarting PyTorch GNN Training...")

for epoch in range(1, EPOCHS + 1):

	train_loss, train_acc = train()
	val_loss, val_acc = test(val_loader)
    
	print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

print("\nTraining complete!")


# --- 7. FINAL VALIDATION PREDICTION AND COMPARISON ---

@torch.no_grad()
def get_full_predictions(loader):
    """Runs the model on a loader and collects all true and predicted labels."""
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    for data in loader:
        # 1. Forward pass
        out = model(data)
        
        # 2. Get predicted class IDs (index of max log-probability)
        pred = out.argmax(dim=1)
        
        # 3. Collect true and predicted labels (PyTorch tensors)
        all_pred_labels.append(pred)
        all_true_labels.append(data.y)

    # Concatenate all batches into single tensors
    true_labels_tensor = torch.cat(all_true_labels, dim=0)
    pred_labels_tensor = torch.cat(all_pred_labels, dim=0)
    
    # Convert to NumPy for easy printing/comparison
    true_labels_np = true_labels_tensor.cpu().numpy()
    pred_labels_np = pred_labels_tensor.cpu().numpy()
    
    return true_labels_np, pred_labels_np


print("\n--- Final Validation Prediction ---")

# Calculate the predictions for the entire validation set
val_true_labels_np, val_pred_labels_np = get_full_predictions(val_loader)

print("\nEncoded val true:", val_true_labels_np)
print("Encoded val pred:", val_pred_labels_np)

# Calculate and print the overall accuracy (sanity check)
final_val_accuracy = np.mean(val_true_labels_np == val_pred_labels_np)
print(f"\nFinal Validation Accuracy: {final_val_accuracy:.4f}")

# ---

	
	
	
	
	
	
	
	
	
	
	
	






























