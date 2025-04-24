# GraphPharma
GraphPharma is a project designed for processing BindingDB data, generating molecular representations, embedding protein sequences, and training a networks for IC50 prediction.

## Project Structure
```
GraPharma/
│
├── best_configs/                  # Saved JSON configs of best hyperparameters for each model
│   ├── best_ffn_config*.json
│   ├── best_gnn_config*.json
│   └── best_lstm_config*.json
│
├── best_models/                   # Trained PyTorch model checkpoints for the best FFN/GNN/LSTM
│   ├── best_ffn*.pth
│   ├── best_gnn*.pth
│   └── best_lstm*.pth
│
├── data/                          # Raw BindingDB downloads
│   ├── BindingDB_All_202503_tsv.zip
│   └── rep_13k.zip                # 13K-sample subset used in training (must unzip to use)
│
├── data_processor/               # Scripts for processing data
│   ├── base_processor.py         # Common utility methods for loading/saving
│   ├── binding_db_processor.py   # Generates fingerprints, embeddings, and graphs
│   ├── protein_embedder.py       # Protein embedding via pretrained transformer
│   ├── visualizer.py             # Optional plotting utilities
│   └── ic50.npy                  # Saved IC50 values
│
├── metrics/                      # Logs and training metrics
│   ├── *_performance_log.csv     # Final performance summary for each model
│   ├── *_train_loss*.npy         # Train loss curves per epoch
│   ├── *_val_loss*.npy           # Val loss curves per epoch
│   └── gnn_with_protein_fix.csv  # GNN run with protein-embedding fix
│
├── networks/                     # Neural network architecture definitions
│   ├── grapharma_ffn.py          # Feedforward NN for IC50
│   ├── grapharma_gnn.py          # GNNs (GCN, SAGE, GAT)
│   └── grapharma_lstm.py         # LSTM, BiLSTM, GRU models
│
├── plots/                        # Saved visualizations from best models
│   ├── best_ffn_training_curve.png
│   ├── best_gnn_training_curve.png
│   ├── best_lstm_training_curve.png
│   └── test_rmse.png
│
├── representations/             # Preprocessed data files for model input
│   ├── bindingdb_processed_embeddings.{csv,npy}
│   ├── bindingdb_processed_fingerprints.{csv,npy}
│   ├── bindingdb_processed_filtered.csv
│   ├── bindingdb_processed_graphs.{csv,pkl}
│   └── ic50.npy
│ 
├── plotfunc.py                   # Helper for plotting metrics
├── ic50Trainer.py                # LSTM and FNN model training pipeline
├── graphIC50Trainer.py           # GNN model training pipeline
├── train_ffn.py                  # Training script for FFN
├── train_gnn.py                  # Training script for GNN
├── train_lstm.py                 # Training script for LSTM
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md                     # Project documentation

```

## Setup
Make sure you have Python 3.13+ installed and set up a virtual environment:

```python
python3 -m venv .venv

source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

##  Usage

### Generate Representations
To generate molecular fingerprints, protein embeddings, and graph representations, run:

```python
python3 binding_db_processor.py
```
This will process BindingDB data and save the outputs in the `representations/` folder.


### Training

Each script below handles training, logs performance, and saves best models/configs.

### Train FFN
```bash
python3 train_ffn.py
```

### Train GNN (GCN, GAT, GraphSAGE variants)
```bash
python3 train_gnn.py
```

### Train LSTM / BiLSTM / GRU
```bash
python3 train_lstm.py
```

### Logs & Visualizations

- Performance logs: `metrics/*.csv`
- Training curves: `plots/*.png`
- Best model configs: `best_configs/*.json`
- Best PyTorch models: `best_models/*.pth`


