import torch
import esm
class ProteinEmbedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # disable dropout for deterministic results

    def embed_sequence(self, sequence):
        data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        token_representations = results["representations"][33]
        sequence_representation = token_representations[0, 1:len(sequence)+1].mean(0)  # mean-pooled
        return sequence_representation.numpy()
