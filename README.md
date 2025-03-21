# GraphPharma
GraphPharma is a project designed for processing BindingDB data, generating molecular representations, embedding protein sequences, and training a networks for IC50 prediction.

## Project Structure
```
GRAPHARMA/
│── data/                           # Stores raw data
│   ├── BindingDB_All_202503_tsv.zip
│   ├── BindingDB_All.tsv
│
│── representations/                 # Stores processed representations
│   ├── bindingdb_processed_embeddings.csv
│   ├── bindingdb_processed_embeddings.npy
│   ├── bindingdb_processed_filtered.csv
│   ├── bindingdb_processed_fingerprints.csv
│   ├── bindingdb_processed_fingerprints.npy
│   ├── bindingdb_processed_graphs.csv
│   ├── bindingdb_processed_graphs.pkl
│   ├── ic50.npy
│
│── base_processor.py                 # Superclass with saving functions
│── binding_db_processor.py           # Generates molecular representations & saves them
│── graphpharma_nn.py                 # Simple feedforward neural network (FFN)
│── logger.py                          # Handles logging
│── main.py                            # Runs model training
│── protein_embedder.py                # Embeds protein target sequences
│── visualizer.py                      # Visualization & plotting functions
│── README.md                          # Project documentation
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


### Train the Neural Network
Once the representations are generated, train the Feedforward Neural Network (FFN):

```python
python3 main.py
```
