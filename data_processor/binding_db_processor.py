import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import networkx as nx

from base_processor import BaseProcessor
from protein_embedder import ProteinEmbedder


from rdkit import RDLogger
from data_processor.visualizer import GraPharmaVisualizer

RDLogger.DisableLog('rdApp.warning')
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import logger



class BindingDBProcessor(BaseProcessor):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        logger.logger.info(f"[\u2713] BindingDBProcessor Initialized")
     

    def load_data(self, tsv_filename="BindingDB_All.tsv",max_rows=1000):
        """
        Extracts and loads the BindingDB dataset from a ZIP archive, handling bad lines.
        """
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            z.extract(tsv_filename, path="./data/")
        logger.logger.info(f"[\u2713] Dataset Extracted")

        try:
            self.data = pd.read_csv(f"./data/{tsv_filename}", sep="\t", low_memory=False, on_bad_lines="skip", nrows=max_rows)
            logger.logger.info(f"[\u2713] Dataset Loaded: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        except pd.errors.ParserError as e:
            logger.logger.error(f"[!] Parsing error encountered: {e}")


    def filter_ic50_and_extract_sequences(self):
     
        if self.data is None:
            logger.logger.error(f"[!] Dataset not loaded. Run load_data() first.")
            return

        ic50_column = None
        for col in self.data.columns:
            if "IC50" in col:
                ic50_column = col
                break

        if not ic50_column:
            logger.logger.error(f"[!] No IC50 column found in dataset.")
            return


        required_columns = ['Ligand SMILES', 'Target Name', ic50_column, 'BindingDB Target Chain Sequence']
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            logger.logger.error(f"[!] Missing required columns: {missing}")
            return

   
        self.data = self.data[required_columns].dropna()
        self.data[ic50_column] = pd.to_numeric(self.data[ic50_column], errors='coerce')
        self.data = self.data.dropna()

   
        self.data.rename(columns={
            ic50_column: "IC50_nM",
            'BindingDB Target Chain Sequence': "Protein_Sequence"
        }, inplace=True)

        if 'UniProt (SwissProt) Primary ID of Target Chain' in self.data.columns:
            self.data.rename(columns={'UniProt (SwissProt) Primary ID of Target Chain': "UniProt_ID"}, inplace=True)

        logger.logger.info(f"[\u2713] Filtered {self.data.shape[0]} samples with IC50 and protein sequence data.")
        logger.logger.info(f"[\u2713] Found {self.data['Protein_Sequence'].nunique()} unique protein sequences.")
       
    
    def convert_smiles_to_fingerprints(self):
      
            if self.data is None or "Ligand SMILES" not in self.data.columns:
                logger.logger.error(f"[!] Dataset not loaded or filtered. Run filter_ic50_data() first.")
                return

            fingerprints = []
            for smi in tqdm(self.data["Ligand SMILES"].dropna(),desc="\033[36mGenerating Fingerprints\033[0m"):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints.append(fp.ToList())
                else:
                    fingerprints.append(None)
            
            self.data["Fingerprint"] = fingerprints
            self.data.dropna(subset=["Fingerprint"], inplace=True)
            logger.logger.info(f"[\u2713] Generated {len(fingerprints)} molecular fingerprints.")



    def generate_protein_embeddings(self):
     
        if "Protein_Sequence" not in self.data.columns:
            logger.logger.error(f"[!] Dataset not loaded or filtered. Run filter_ic50_data() first.")
            return
        
        embedder = ProteinEmbedder()
        embeddings = []
        
        for seq in tqdm(self.data["Protein_Sequence"],desc="\033[36mGenerating Protein Embeddings\033[0m"):
            try:
                emb = embedder.embed_sequence(seq)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error embedding sequence: {e}")
                embeddings.append(None)

        self.data["Protein_Embedding"] = embeddings
        self.data.dropna(subset=["Protein_Embedding"], inplace=True)
        logger.logger.info(f"[\u2713] Generated embeddings for {len(self.data)} sequences.")

    def generate_graph_representation(self):
         
            if self.data is None or "Ligand SMILES" not in self.data.columns:
                logger.logger.error(f"[!] Dataset not loaded or filtered. Run filter_ic50_data() first.")

                return
            
            graphs = []
            for smi in tqdm(self.data["Ligand SMILES"].dropna(),desc="\033[36mGraph Representations\033[0m"):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    G = nx.Graph()
                    for atom in mol.GetAtoms():
                        G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
                    for bond in mol.GetBonds():
                        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=bond.GetBondType())
                    graphs.append(G)
            
            self.data["Graph_Representation"] = graphs
            logger.logger.info(f"[\u2713] Generated {len(graphs)} graph-based representations.")

      
if __name__=="__main__":

    processor = BindingDBProcessor("../data/BindingDB_All_202503_tsv.zip")
    processor.load_data()

    processor.filter_ic50_and_extract_sequences()

    processor.convert_smiles_to_fingerprints()
    processor.generate_protein_embeddings()
    processor.generate_graph_representation()

    visualizer=GraPharmaVisualizer(processor.data)
    visualizer.run_visualizations()
    processor.save()

    
