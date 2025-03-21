import pickle
import logger
import numpy as np
import pandas as pd

class BaseProcessor:
    def __init__(self,zip_path):
    
        self.zip_path = zip_path
        self.data = None

    def save(self, prefix="./representations/bindingdb_processed"):
        self._save_filtered_data(f"{prefix}_filtered.csv")
        self._save_fingerprints(f"{prefix}_fingerprints.npy")
        self._save_protein_embeddings(f"{prefix}_embeddings.npy")
        self._save_graphs(f"{prefix}_graphs.pkl")
        self._save_fingerprints_csv(filename=f"{prefix}_fingerprints.csv")
        self._save_protein_embeddings_csv(filename=f"{prefix}_embeddings.csv")
        self._save_graphs_csv(filename=f"{prefix}_graphs.csv")

    def _save_filtered_data(self, filename="BindingDB_IC50_filtered.csv"):
        """
        Saves the filtered data to a CSV file (including IC50, SMILES, sequence, etc.).
        """
        if self.data is None:
            logger.logger.error(f"[!] No data to save.")
            
            return
        self.data.to_csv(filename, index=False)
        logger.logger.info(f"[\u2713] Filtered data saved to {filename}")


    def _save_fingerprints(self, filename="fingerprints.npy", ic50_file="ic50.npy"):
        """
        Saves molecular fingerprints as a NumPy array.
        """
        if "Fingerprint" not in self.data.columns:
            logger.logger.error(f"[!] Fingerprints not generated. Run convert_smiles_to_fingerprints() first.")
            return
        np.save(filename, self.data["Fingerprint"].to_list())
        np.save(ic50_file, self.data["IC50_nM"].to_numpy())
        logger.logger.info(f"[\u2713] Fingerprints saved to {filename}")
        logger.logger.info(f"[\u2713] IC50 values saved to {ic50_file}")

    def _save_fingerprints_csv(self, filename="fingerprints.csv"):
        """
        Saves fingerprints and IC50 to a CSV file.
        """
        if "Fingerprint" not in self.data.columns or "IC50_nM" not in self.data.columns:
            logger.logger.error(f"[!] Data not ready. Make sure fingerprints and IC50 exist.")
            return
        
        df = pd.DataFrame(self.data["Fingerprint"].to_list())
        df["IC50_nM"] = self.data["IC50_nM"].values
        df.to_csv(filename, index=False)
        logger.logger.info(f"[\u2713] Fingerprints and IC50 saved to {filename}")

    def _save_protein_embeddings(self, filename="protein_embeddings.npy"):
        """
        Saves protein embeddings as a NumPy array.
        """
        if "Protein_Embedding" not in self.data.columns:
            logger.logger.error(f"[!] Embeddings not found. Run generate_protein_embeddings() first.")
            return
        np.save(filename, self.data["Protein_Embedding"].to_list())
        logger.logger.info(f"[\u2713] Protein embeddings saved to {filename}")

    def _save_protein_embeddings_csv(self, filename="protein_embeddings.csv"):
        """
        Saves protein embeddings to a CSV file.
        """
        if "Protein_Embedding" not in self.data.columns:
            logger.logger.error(f"[!] Protein embeddings not found.")
            return

        df = pd.DataFrame(self.data["Protein_Embedding"].to_list())
        df.to_csv(filename, index=False)
        logger.logger.info(f"[\u2713] Protein embeddings saved to {filename}")

    def _save_graphs(self, filename="graph_representations.pkl"):
        """
        Saves graph representations to a pickle file.
        """
        if "Graph_Representation" not in self.data.columns:
            logger.logger.error(f"[!] Graph representations not found. Run generate_graph_representation() first.")

            return
        with open(filename, "wb") as f:
            pickle.dump(self.data["Graph_Representation"].to_list(), f)
        logger.logger.info(f"[\u2713] Graph representations saved to {filename}")
        
    def _save_graphs_csv(self, filename="graph_representations.csv"):
        """
        Saves graph representations to a CSV file (as strings).
        """
        if "Graph_Representation" not in self.data.columns:
            logger.logger.error(f"[!] Graph representations not found.")
            return

        df = pd.DataFrame({
            "Graph_Representation": [str(graph) for graph in self.data["Graph_Representation"].to_list()]
        })
        df.to_csv(filename, index=False)
        logger.logger.info(f"[\u2713] Graph representations saved to {filename}")
