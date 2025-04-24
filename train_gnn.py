# train_gnn.py
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from networks.grapharma_gnn import GNN
from graphIC50Trainer import GraphIC50Trainer

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'H', 'B', 'Si', 'Se', 'Zn', 'Mg', 'Fe', 'Ca', 'Cu',
    'Na', 'K', 'Mn', 'Al', 'Co', 'Ni', 'V', 'Cr', 'Ti',
]

if __name__ == "__main__":
    graphs = pickle.load(open("./data/rep_13k/bindingdb_processed_graphs.pkl", "rb"))
    ic50 = np.load("./data/rep_13k/bindingdb_processed_ic50.npy")
    ic50 = np.log1p(ic50).astype(np.float32)

    X = [g for g in graphs if g.number_of_nodes() > 0]
    y = [ic50[i] for i in range(len(graphs)) if graphs[i].number_of_nodes() > 0]

    trainer = GraphIC50Trainer()
    trainer.preprocess(X, y, batch_size=32)

    headers = ['architecture', 'hidden_channels', 'learning_rate', 'num_layers', 'train_loss', 'val_loss', 'test_rmse']
    log_file = "./metrics/gnn_performance_log1.csv"
    trainer.init_log(log_file, headers=headers)

    best_test_rmse = float('inf')
    best_config = None

    hidden_channel_options = [16, 32, 64]
    learning_rate_options = [0.0001, 0.001, 0.01, 0.1]
    gnn_types = ['GCN', 'SAGE', 'GAT']
    num_layers_options = [2, 3, 4]
    epochs = 50

    for hidden_channels in hidden_channel_options:
        for lr in learning_rate_options:
            for gnn_type in gnn_types:
                for num_layers in num_layers_options:
                    print(f"Training with: hidden_channels={hidden_channels}, lr={lr}, gnn_type={gnn_type}, num_layers={num_layers}")

                    model = GNN(in_channels=len(ATOM_TYPES), hidden_channels=hidden_channels, num_layers=num_layers, gnn_type=gnn_type)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    trained_model, train_losses, val_losses = trainer.train(model, epochs, optimizer, criterion)
                    test_rmse, _, _ = trainer.test(trained_model, verbose=False)

                    print(f"Final Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Test RMSE: {test_rmse:.4f}")

                    log_row = [gnn_type, hidden_channels, lr, num_layers,
                               round(train_losses[-1], 4), round(val_losses[-1], 4), round(test_rmse, 4)]
                    trainer.log_results(log_row)

                    if test_rmse < best_test_rmse:
                        best_test_rmse = test_rmse
                        best_config = {
                            'gnn_type': gnn_type,
                            'hidden_channels': hidden_channels,
                            'lr': float(lr),
                            'num_layers': num_layers,
                            'test_rmse': round(float(test_rmse), 4)
                        }
                        torch.save(trained_model.state_dict(), "./best_models/best_gnn1.pth")
                        np.save("./metrics/gnn_train_loss1.npy", train_losses)
                        np.save("./metrics/gnn_val_loss1.npy", val_losses)

    if best_config:
        with open("./best_configs/best_gnn_config1.json", "w") as f:
            json.dump(best_config, f, indent=4)

    print(f"\nAll results logged to {log_file}")

    # Plot best model's training and validation loss
    train_loss = np.load("./metrics/gnn_train_loss1.npy")
    val_loss = np.load("./metrics/gnn_val_loss1.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Best GNN Model: Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/best_gnn_training_curve.png", dpi=300)
    plt.show()
