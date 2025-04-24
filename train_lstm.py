import numpy as np
import torch
import torch.nn as nn
from networks.grapharma_lstm import LSTM_IC50
from ic50Trainer import IC50Trainer
import json
import matplotlib.pyplot as plt

if __name__=="__main__":

    fingerprints = np.load("./data/rep_13k/bindingdb_processed_fingerprints.npy", allow_pickle=True)
    embeddings = np.load("./data/rep_13k/bindingdb_processed_embeddings.npy", allow_pickle=True)
    ic50 = np.load("./data/rep_13k/bindingdb_processed_ic50.npy")


    fingerprints = np.array(fingerprints.tolist())
    embeddings = np.array(embeddings.tolist())
    ic50 = np.log1p(ic50).reshape(-1, 1).astype(np.float32)


    if embeddings.shape[1] < fingerprints.shape[1]:
        pad_width = fingerprints.shape[1] - embeddings.shape[1]
        embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), 'constant')

    X = np.stack([fingerprints, embeddings], axis=1)
    y = ic50


    batch_size=64
    trainer=IC50Trainer()
    trainer.preprocess(X, y)
    log_file = "./metrics/lstm_performance_log.csv"
    headers = ['model_type', 'hidden_dim', 'num_layers', 'architecture', 'dropout', 'learning_rate', 'train_loss', 'val_loss', 'test_rmse']

    trainer.init_log(log_file,headers)

    model_types = ['lstm', 'bilstm', 'gru']
    architectures = [
        [512, 128],
        [1024, 512, 128],
        [256, 64]
    ]
    dropouts = [0.1, 0.2]
    lrs = [1e-3, 1e-4]
    hidden_dims = [64, 128, 256]
    num_layers_list = [1, 2]
    epochs = 50
    best_test_rmse = float('inf') 
    best_config = None  



    for model_type in model_types:
        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                for arch in architectures:
                    for dropout in dropouts:
                        for lr in lrs:
                            print(f"\nModel: {model_type.upper()}, Hidden: {hidden_dim}, Layers: {num_layers}, Arch: {arch}, Dropout: {dropout}, LR: {lr}")
                            model = LSTM_IC50(
                                input_dim=X.shape[2],
                                hidden_dim=hidden_dim,
                                architecture=arch,
                                dropout=dropout,
                                num_layers=num_layers,
                                model_type=model_type
                            )
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            criterion = nn.MSELoss()

                            trained_model, train_loss, val_loss = trainer.train(model,epochs, optimizer, criterion)
                            test_rmse, _, _, _ = trainer.test(trained_model, verbose=False)

                            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test RMSE: {test_rmse:.4f}")
                            log_row = [
                                model_type,
                                hidden_dim,
                                num_layers,
                                str(arch),
                                dropout,
                                lr,
                                round(train_loss, 4),
                                round(val_loss, 4),
                                round(test_rmse, 4)
                            ]
                            trainer.log_results(log_row)
                            if test_rmse < best_test_rmse:
                                best_test_rmse = test_rmse
                                best_config = {
                                    'lstm_type': model_type,
                                    'hidden_dimensions': hidden_dim,
                                    'architecture': str(arch),
                                    "dropout":dropout,
                                    'lr': lr,
                                    'num_layers': num_layers,
                                    'test_rmse':   round(float(test_rmse), 4) 
  
                                }
                            
                                torch.save(trained_model.state_dict(), "./best_models/best_lstm1.pth")
                                np.save("./metrics/lstm_train_loss1.npy", train_loss)
                                np.save("./metrics/lstm_val_loss1.npy", val_loss)



    if best_config:
        with open("./best_configs/best_lstm_config1.json", "w") as f:
            json.dump(best_config, f, indent=4)

    print(f"\nAll results logged to {log_file}")

    # Plot best model's training and validation loss
    train_loss = np.load("./metrics/lstm_train_loss1.npy")
    val_loss = np.load("./metrics/lstm_val_loss1.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Best LSTM Model: Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/best_lstm_training_curve.png", dpi=300)
    plt.show()