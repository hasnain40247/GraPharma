import matplotlib.pyplot as plt
import numpy as np
from logger import logger
import networkx as nx
import sklearn
from sklearn.manifold import TSNE

class GraPharmaPlots:
    def __init__(self, data):
        self.data = data

    def plot_ic50_distribution(self, log_scale=True):
        if self.data is None:
            logger.error("[!] Dataset not loaded. Cannot plot IC50 distribution.")
            return

        plt.figure(figsize=(8, 5))
        if log_scale:
            plt.hist(np.log10(self.data["IC50_nM"]), bins=50, color="blue", alpha=0.7)
            plt.xlabel("Log10(IC50) (nM)")
        else:
            plt.hist(self.data["IC50_nM"], bins=50, color="blue", alpha=0.7)
            plt.xlabel("IC50 (nM)")

        plt.ylabel("Frequency")
        plt.title("IC50 Distribution")
        plt.grid(True)
        plt.show()

    def plot_fingerprint_density(self):
        if "Fingerprint" not in self.data.columns:
            logger.error("[!] Fingerprints not available.")
            return

        arr = np.array(self.data["Fingerprint"].to_list())
        density = arr.mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(density)
        plt.title("Fingerprint Bit Activation Density")
        plt.xlabel("Bit Index")
        plt.ylabel("Activation Frequency")
        plt.grid(True)
        plt.show()

    def plot_top_targets(self, top_n=10):
        if "Target Name" not in self.data.columns:
            logger.error("[!] Target Name column not found.")
            return

        target_counts = self.data["Target Name"].value_counts().nlargest(top_n)
        target_counts.plot(kind="barh", color="teal", figsize=(8, 5))
        plt.xlabel("Number of Interactions")
        plt.title(f"Top {top_n} Target Proteins")
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()

    def plot_sequence_length_distribution(self):
        if "Protein_Sequence" not in self.data.columns:
            logger.error("[!] Protein_Sequence column not found.")
            return

        lengths = self.data["Protein_Sequence"].apply(len)
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=50, color="purple", alpha=0.7)
        plt.xlabel("Protein Sequence Length")
        plt.ylabel("Frequency")
        plt.title("Protein Sequence Length Distribution")
        plt.grid(True)
        plt.show()

    def plot_ic50_vs_sequence_length(self):
        if "Protein_Sequence" not in self.data.columns or "IC50_nM" not in self.data.columns:
            logger.error("[!] Required columns not found.")
            return

        lengths = self.data["Protein_Sequence"].apply(len)
        plt.figure(figsize=(8, 5))
        plt.scatter(lengths, np.log10(self.data["IC50_nM"]), alpha=0.6)
        plt.xlabel("Protein Sequence Length")
        plt.ylabel("Log10(IC50)")
        plt.title("IC50 vs Protein Sequence Length")
        plt.grid(True)
        plt.show()

    def plot_fingerprints(self, method='tsne', sample_size=500):
        if "Fingerprint" not in self.data.columns:
            logger.error("[!] Molecular fingerprints not found.")
            return

        fps = np.array(self.data["Fingerprint"].tolist())
        
        if len(fps) > sample_size:
            idx = np.random.choice(len(fps), sample_size, replace=False)
            fps = fps[idx]

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        else:
            logger.error("[!] Unsupported reduction method.")
            return

        reduced = reducer.fit_transform(fps)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, color='green')
        plt.title(f"Fingerprint Vectors ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.show()
    def plot_protein_embeddings(self, method='tsne', perplexity=30):
        if "Protein_Embedding" not in self.data.columns:
            logger.error("[!] Protein embeddings not found.")
            return
        
        embeddings = np.vstack(self.data["Protein_Embedding"])
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        else:
            logger.error("[!] Unsupported reduction method.")
            return

        reduced = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        plt.title(f"Protein Embeddings ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.show()

    def plot_molecule_graphs(self, num_graphs=5):
        if "Graph_Representation" not in self.data.columns:
            logger.error("[!] Graph representations not found in dataset.")
            return

        sampled_graphs = self.data["Graph_Representation"].dropna().sample(min(num_graphs, len(self.data)))
        
        for i, G in enumerate(sampled_graphs, 1):
            plt.figure(figsize=(4, 4))
            pos = nx.spring_layout(G, seed=42)  # consistent layout
            node_labels = nx.get_node_attributes(G, 'symbol')
            nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='skyblue', edge_color='gray', node_size=800, font_size=10)
            plt.title(f"Molecule Graph #{i}")
            plt.tight_layout()
            plt.show()


class GraPharmaVisualizer:
    def __init__(self, data):
        self.data = data
        self.plots = GraPharmaPlots(data)

    def run_visualizations(self):
        self.plots.plot_ic50_distribution()
        self.plots.plot_sequence_length_distribution()
        self.plots.plot_ic50_vs_sequence_length()
        self.plots.plot_top_targets()
        self.plots.plot_fingerprint_density()
        self.plots.plot_protein_embeddings()
        self.plots.plot_fingerprints()
        self.plots.plot_molecule_graphs(num_graphs=3)