from ic50Trainer import IC50Trainer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'P',  # Basic organic atoms
    'F', 'Cl', 'Br', 'I',     # Halogens
    'H', 'B', 'Si', 'Se', 'Zn', 'Mg', 'Fe', 'Ca', 'Cu',  # Less common in small molecules but sometimes present
    'Na', 'K', 'Mn', 'Al', 'Co', 'Ni', 'V', 'Cr', 'Ti',  # Metals and trace elements
]

class GraphIC50Trainer(IC50Trainer):
    def __init__(self):
        super().__init__()
        


    def atom_symbol_to_one_hot(self,symbol):
        """
        Converts atom symbol to one-hot encoding.
        """
        one_hot = [0] * len(ATOM_TYPES)
        if symbol in ATOM_TYPES:
            one_hot[ATOM_TYPES.index(symbol)] = 1
        return one_hot

    def nx_to_pyg_data(self,graph, label):
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
            one_hot = self.atom_symbol_to_one_hot(symbol)
            node_features.append(one_hot)

        # Convert to tensor [num_nodes, num_features]
        x = torch.tensor(node_features, dtype=torch.float)
        # Create PyTorch Geometric Data object
        return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

    
    def preprocess(self, X, y, test_size=0.2, val_size=0.1, random_state=42, scale=False, batch_size=16):
        # First split into test and temp sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Now split the temp set into train and validation sets, using val_size as a proportion of the temp set
        val_ratio = val_size / (1 - test_size)  # adjust the validation size relative to the remaining data
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)

        # Create Pytorch Geometric data objects
        train_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_train, y_train)]
        val_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_val, y_val)]
        test_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_test, y_test)]

        # Filter out any potential Nones
        train_data = [data for data in train_data if data is not None]
        val_data = [data for data in val_data if data is not None]
        test_data = [data for data in test_data if data is not None]

        # Create DataLoader instances
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    def train(self,model, epochs, optimizer, criterion):

        """
        Training function for GNN with validation loss.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data in self.train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                preds = model(data.x, data.edge_index, data.batch)
                loss = criterion(preds, data.y.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
         
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in self.val_loader:
                    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
                    preds = model(data.x, data.edge_index, data.batch)
                    loss = criterion(preds, data.y.view(-1))
                    val_loss += loss.item()
     

            avg_train_loss = running_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        return model, avg_train_loss, avg_val_loss


    def test(self,model,verbose=True):

        """
        Testing function for GNN.
        """
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
                pred = model(data.x, data.edge_index, data.batch)
                
                preds.extend(pred.cpu().numpy())
                trues.extend(data.y.cpu().numpy())

        all_preds = np.array(preds).squeeze()
        all_labels = np.array(trues).squeeze()
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))

        # rmse = np.sqrt(np.mean((preds - trues) ** 2))
   
        if verbose:
            print(f"\nTest RMSE (log IC50): {rmse:.4f}\n")
            print("Predictions on Test Data:\n")
            for i in range(len(preds[:10])):
            
                pred_log_ic50 = all_preds[i]
                true_log_ic50 = all_labels[i]
                pred_ic50 = np.expm1(pred_log_ic50)
                true_ic50 = np.expm1(true_log_ic50)

                print(f"Sample {i+1}")
            
                print(f"Predicted log(IC50): {pred_log_ic50:.4f}")
                print(f"True log(IC50):      {true_log_ic50:.4f}")
                print(f"Predicted IC50:      {pred_ic50:.2f} nM")
                print(f"True IC50:           {true_ic50:.2f} nM")
                print("-" * 80)
        return rmse, all_preds, all_labels

