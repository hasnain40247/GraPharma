# graphIC50Trainer.py
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from ic50Trainer import IC50Trainer

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'H', 'B', 'Si', 'Se', 'Zn', 'Mg', 'Fe', 'Ca', 'Cu',
    'Na', 'K', 'Mn', 'Al', 'Co', 'Ni', 'V', 'Cr', 'Ti',
]

class GraphIC50Trainer(IC50Trainer):
    def __init__(self):
        super().__init__()

    def atom_symbol_to_one_hot(self, symbol):
        one_hot = [0] * len(ATOM_TYPES)
        if symbol in ATOM_TYPES:
            one_hot[ATOM_TYPES.index(symbol)] = 1
        return one_hot

    def nx_to_pyg_data(self, graph, label):
        if graph.number_of_nodes() == 0:
            return None
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        atom_symbols = nx.get_node_attributes(graph, 'symbol')
        node_features = [self.atom_symbol_to_one_hot(symbol) for _, symbol in atom_symbols.items()]
        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

    def preprocess(self, X, y, test_size=0.2, val_size=0.1, random_state=42, scale=False, batch_size=16):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)

        train_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_train, y_train)]
        val_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_val, y_val)]
        test_data = [self.nx_to_pyg_data(g, label) for g, label in zip(X_test, y_test)]

        self.train_loader = DataLoader([d for d in train_data if d], batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader([d for d in val_data if d], batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader([d for d in test_data if d], batch_size=batch_size, shuffle=False)

    def train(self, model, epochs, optimizer, criterion):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        train_loss_per_epoch = []
        val_loss_per_epoch = []

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
                    data = data.to(device)
                    preds = model(data.x, data.edge_index, data.batch)
                    loss = criterion(preds, data.y.view(-1))
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)

            train_loss_per_epoch.append(avg_train_loss)
            val_loss_per_epoch.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        return model, train_loss_per_epoch, val_loss_per_epoch

    def test(self, model, verbose=True):
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

        if verbose:
            print(f"\nTest RMSE (log IC50): {rmse:.4f}\n")
            for i in range(min(10, len(all_preds))):
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
