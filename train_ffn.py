

import numpy as np
import torch
import torch.nn as nn
from networks.grapharma_nn import GraPharmaNN
from ic50Trainer import IC50Trainer
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fingerprint_file = "./data/rep_13k/bindingdb_processed_fingerprints.npy"
    ic50_file = "./data/rep_13k/bindingdb_processed_ic50.npy"
    epochs = 50
    batch_size = 64
    trainer=IC50Trainer()

    X = np.load(fingerprint_file, allow_pickle=True)
    y = np.load(ic50_file)
    X = np.array(X.tolist())  
    y = np.log1p(y).reshape(-1, 1).astype(np.float32)

    trainer.preprocess(X, y,scale=True)
    headers = ['architecture', 'dropout', 'learning_rate', 'train_loss', 'val_loss', 'test_rmse']
    
    log_file = "./metrics/ffn_performance_log1.csv"
    trainer.init_log(log_file,headers=headers)
    best_test_rmse = float('inf') 
    best_config = None 


    architectures = [
        [512, 128],
        [1024, 512, 128],
        [256, 64]
    ]
    dropouts = [0.1, 0.2]
    lrs = [1e-3, 1e-4]

    for arch in architectures:
        for dropout in dropouts:
            for lr in lrs:
                print(f"\nTesting architecture: {arch}, dropout: {dropout}, lr: {lr}")
                model = GraPharmaNN(input_dim=X.shape[1], architecture=arch, dropout=dropout)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                trained_model, train_loss, val_loss = trainer.train(model, epochs, optimizer, criterion)
                test_rmse, _, _, _ = trainer.test(trained_model, verbose=False)

                print(f"Final Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test RMSE: {test_rmse:.4f}")
                   
                log_row = [
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
                                'architecture': str(arch),
                                'dropout': dropout,
                                'lr': lr,
                                'test_rmse':   round(float(test_rmse), 4) 

                            }
                         
                            torch.save(trained_model.state_dict(), "./best_models/best_ffn1.pth")
                            np.save("./metrics/ffn_train_loss1.npy", train_loss)
                            np.save("./metrics/ffn_val_loss1.npy", val_loss)



    if best_config:
        with open("./best_configs/best_ffn_config1.json", "w") as f:
            json.dump(best_config, f, indent=4)

    print(f"\nAll results logged to {log_file}")

    # Plot best model's training and validation loss
    train_loss = np.load("./metrics/ffn_train_loss1.npy")
    val_loss = np.load("./metrics/ffn_val_loss1.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Best FFN Model: Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/best_ffn_training_curve.png", dpi=300)
    plt.show()
