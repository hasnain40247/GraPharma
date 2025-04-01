import numpy as np
import pickle
import torch
import torch.nn as nn
import networkx as nx
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from grapharma_gnn import GNN

graphs = pickle.load(open("./representations/bindingdb_processed_graphs.pkl", "rb"))
ic50 = np.load("./representations/ic50.npy")
ic50 = np.log1p(ic50).astype(np.float32) # Log-transform IC50 values

# Define unique atom types for one-hot encoding
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'P',  # Basic organic atoms
    'F', 'Cl', 'Br', 'I',     # Halogens
    'H', 'B', 'Si', 'Se', 'Zn', 'Mg', 'Fe', 'Ca', 'Cu',  # Less common in small molecules but sometimes present
    'Na', 'K', 'Mn', 'Al', 'Co', 'Ni', 'V', 'Cr', 'Ti',  # Metals and trace elements
]


def atom_symbol_to_one_hot(symbol):
    """
    Converts atom symbol to one-hot encoding.
    """
    one_hot = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        one_hot[ATOM_TYPES.index(symbol)] = 1
    return one_hot


def nx_to_pyg_data(graph, label):
    """
    Converts a NetworkX graph to PyTorch Geometric Data.
    """
    if graph.number_of_nodes() == 0:
        return None
    
    # Get edge list and convert to tensor [2, num_edges]
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    
    # Get atom symbols and convert to one-hot features
    atom_symbols = nx.get_node_attributes(graph, 'symbol')
    node_features = []
    for _, symbol in atom_symbols.items():
        one_hot = atom_symbol_to_one_hot(symbol)
        node_features.append(one_hot)

    # Convert to tensor [num_nodes, num_features]
    x = torch.tensor(node_features, dtype=torch.float)
    # Create PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

# Convert all graphs to PyTorch Geometric Data objects
data_list = [nx_to_pyg_data(graphs[i], ic50[i]) for i in range(len(graphs)) if graphs[i].number_of_nodes() > 0]
data_list = [data for data in data_list if data is not None]
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Create DataLoader objects
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the GNN model
in_channels = len(ATOM_TYPES)  # One-hot vector size
hidden_channels = 32
model = GNN(in_channels, hidden_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, loader, epochs):
    """
    Training function for GNN.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            optimizer.zero_grad()
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            preds = model(data.x, data.edge_index, data.batch)
            loss = criterion(preds, data.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

train(model, train_loader, epochs=50)


def test(model, loader):
    """
    Testing function for GNN.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            pred = model(data.x, data.edge_index, data.batch)
            
            preds.extend(pred.cpu().numpy())
            trues.extend(data.y.cpu().numpy())

    preds = np.array(preds).squeeze()
    trues = np.array(trues).squeeze()
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    print(f"\nTest RMSE (log IC50): {rmse:.4f}")
    return preds, trues

preds, trues = test(model, test_loader)


print("Predictions on Test Data:\n")
for i in range(10):
    pred_log_ic50 = preds[i]
    true_log_ic50 = trues[i]
    pred_ic50 = np.expm1(pred_log_ic50)
    true_ic50 = np.expm1(true_log_ic50)

    print(f"Sample {i+1}")
    print(f"Predicted log(IC50): {pred_log_ic50:.4f}")
    print(f"True log(IC50):      {true_log_ic50:.4f}")
    print(f"Predicted IC50:      {pred_ic50:.2f} nM")
    print(f"True IC50:           {true_ic50:.2f} nM")
    print("-" * 80)
