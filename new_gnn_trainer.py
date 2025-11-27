import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset
from torch_geometric.nn import GATConv, global_mean_pool 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. GLOBAL CONFIGURATION & HYPERPARAMETERS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 80 # Increased epochs for better convergence
LEARNING_RATE = 0.0005 
BATCH_SIZE = 32

# Data Dimensions for the 16-class classification problem
NUM_CLASSES = 16 
INPUT_FEATURE_DIM = 260 # 256D (LSTM Content) + 4D (Topology Context)
HIDDEN_CHANNELS = 32 # Number of features in the hidden layers
DROPOUT_RATE = 0.5

# Set seed for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- 2. DATASET CLASS (Handles the 3200 .pt files) ---
class BlockCipherDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BlockCipherDataset, self).__init__(root, transform, pre_transform)
        # Load the consolidated data.pt file if it exists
        self.data, self.slices = torch.load(self.processed_paths[0] , weights_only=False)
        
        # NOTE: This is the fix for the security/compatibility error
        # Although commented, remember to handle any remaining security issues 
        # that block the PyG data object loading in your environment setup.
        # torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])


    @property
    def raw_file_names(self):
        # Recursively find all individual .pt files in all class subdirectories
        file_list = []
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith(".pt"):
                        file_list.append(os.path.join(class_folder, filename))
        return file_list

    @property
    def processed_file_names(self):
        return ['data.pt'] # The single, consolidated file used for training

    def process(self):
        data_list = []
        for filename_relative in self.raw_file_names:
            full_path = os.path.join(self.root, filename_relative)
            
            try:
                # CRITICAL: Use weights_only=False to bypass the security block for trusted files
                loaded_dict = torch.load(full_path, weights_only=False) 
                
                # The data object is expected to be stored under the 'pyg_data' key
                data = loaded_dict.get('pyg_data')
                
                if data is None:
                    print(f"Skipping file {full_path}: 'pyg_data' key not found.")
                    continue

                # Feature dimension validation
                if data.x.shape[1] != INPUT_FEATURE_DIM:
                     print(f"ERROR: Feature dim mismatch in {full_path}. Expected {INPUT_FEATURE_DIM}, got {data.x.shape[1]}")
                     continue
                
                data_list.append(data)
            except Exception as e:
                print(f"Skipping file {full_path} due to load error: {e}")

        if not data_list:
            raise IndexError("Data list is empty. All .pt files failed to load. Check the error messages above.")
            
        print(f"Loaded {len(data_list)} total graphs.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# --- 3. GNN MODEL (Graph Attention Network - GAT) ---

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, heads=4):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        # GAT Layer 1: Attends over neighbors, multiplies feature space by 'heads'
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        
        # GAT Layer 2: Aggregates the attention heads back down to a clean feature space
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout_rate)
        
        # Final Linear Layer for 16-class classification
        self.lin = nn.Linear(hidden_channels, out_channels) 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU (Exponential Linear Unit) is often used with GAT
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # GAT Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global Pooling (Readout Layer): Aggregate node features into a single graph feature vector
        x = global_mean_pool(x, batch) 
        
        # Final Logits
        x = self.lin(x)
        return x


# --- 4. TRAINING AND EVALUATION FUNCTIONS ---

def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs
    
    return total_loss / total_samples, correct / total_samples

def evaluate(model, loader, criterion, return_labels=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data)
            loss = criterion(out, data.y) 
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_samples += data.num_graphs
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    if return_labels:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    return avg_loss, accuracy


# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 5.1 Load and Split Data
    DATASET_ROOT = 'dataset' # Folder containing the 16 subdirectories (e.g., 'AES-ECB')
    print(f"Attempting to load data from: {DATASET_ROOT}")
    dataset = BlockCipherDataset(root=DATASET_ROOT)
    
    # Split Indices (Stratified for balanced classes)
    data_indices = np.arange(len(dataset))
    labels = dataset.data.y.numpy()

    # 70% Train, 30% Temp
    train_idx, temp_idx, _, _ = train_test_split(
        data_indices, labels, test_size=0.30, random_state=SEED, stratify=labels)
    
    # 15% Validation, 15% Test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, labels[temp_idx], test_size=0.50, random_state=SEED, stratify=labels[temp_idx])
    
    print(f"\n--- Data Split Summary ---")
    print(f"Total samples: {len(dataset)} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 5.2 Initialize Model, Optimizer, and Loss
    model = GNN(
        in_channels=INPUT_FEATURE_DIM, 
        hidden_channels=HIDDEN_CHANNELS, 
        out_channels=NUM_CLASSES, 
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    criterion = torch.nn.CrossEntropyLoss()

    # 5.3 Training Loop
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Consider saving the model here: torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch: {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} {'*' if val_acc == best_val_acc else ''}")

    # 5.4 Final Evaluation
    print("\n--- Final Test Set Evaluation ---")
    _, final_test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, return_labels=True)
    
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    
    # Print the detailed report showing F1-scores for each of the 16 classes
    print("\n--- Detailed Classification Report ---")
    # You may need to provide target names (cipher names) if your labels aren't integers 0-15
    print(classification_report(test_labels, test_preds, zero_division=0))
    
    # Print confusion matrix for visual assessment
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(test_labels, test_preds))
