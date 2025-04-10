import numpy as np
import pickle
import torch.nn as nn
import torch.optim as optim
from networks.grapharma_gnn import GNN
from graphIC50Trainer import GraphIC50Trainer
import torch
import json
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'P',  # Basic organic atoms
    'F', 'Cl', 'Br', 'I',     # Halogens
    'H', 'B', 'Si', 'Se', 'Zn', 'Mg', 'Fe', 'Ca', 'Cu',  # Less common in small molecules but sometimes present
    'Na', 'K', 'Mn', 'Al', 'Co', 'Ni', 'V', 'Cr', 'Ti',  # Metals and trace elements
]


if __name__=="__main__":
    graphs = pickle.load(open("./representations/bindingdb_processed_graphs.pkl", "rb"))
    ic50 = np.load("./representations/ic50.npy")
    ic50 = np.log1p(ic50).astype(np.float32) 

    X = [graph for graph in graphs if graph.number_of_nodes() > 0]
    y = [ic50[i] for i in range(len(graphs)) if graphs[i].number_of_nodes() > 0]


    batch_size = 32
    trainer=GraphIC50Trainer()
    trainer.preprocess(X,y,batch_size=batch_size)
    headers = ['architecture', 'hidden_channels', 'learning_rate',"num_layers", 'train_loss', 'val_loss', 'test_rmse']
        
    log_file = "./metrics/gnn_performance_log.csv"
    trainer.init_log(log_file,headers=headers)

    in_channels = len(ATOM_TYPES)  
    hidden_channels = 32

    hidden_channel_options = [16, 32, 64]
    learning_rate_options = [0.001, 0.01, 0.1]
    batch_size_options = [16, 32, 64]
    gnn_types = ['GCN', 'SAGE', 'GAT']
    num_layers_options = [2, 3, 4]
    epochs=50

    best_test_rmse = float('inf')  
    best_config = None  



    for hidden_channels in hidden_channel_options:
        for lr in learning_rate_options:
                for gnn_type in gnn_types:
                    for num_layers in num_layers_options:
                        print(f"Training with: hidden_channels={hidden_channels}, lr={lr}, gnn_type={gnn_type}, num_layers={num_layers}")
                        
                        model = GNN(in_channels=len(ATOM_TYPES), hidden_channels=hidden_channels, num_layers=num_layers, gnn_type=gnn_type)
                        criterion = nn.MSELoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    

                        trained_model, train_loss, val_loss = trainer.train(model, epochs, optimizer, criterion)
                        test_rmse, _, _, = trainer.test(trained_model, verbose=False)

                        print(f"Final Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test RMSE: {test_rmse:.4f}")
                    
                        log_row = [
                        str(gnn_type),
                        hidden_channels,
                        lr,
                        num_layers,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(test_rmse, 4)
                                ]
                        trainer.log_results(log_row)

                        if test_rmse < best_test_rmse:
                            best_test_rmse = test_rmse
                            best_config = {
                                'gnn_type': gnn_type,
                                'hidden_channels': hidden_channels,
                                'lr': float(lr),
                                'num_layers': num_layers,
                                'test_rmse':   round(float(test_rmse), 4) 
                            }
                         
                            torch.save(trained_model.state_dict(), "./best_models/best_gnn.pth")



    if best_config:
            with open("./best_configs/best_gnn_config.json", "w") as f:
                json.dump(best_config, f, indent=4)

    print(f"\nAll results logged to {log_file}")
