# import torch
# import esm
# class ProteinEmbedder:
#     def __init__(self, model_name="esm2_t33_650M_UR50D"):
#         self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.model.eval()  # disable dropout for deterministic results

#     def embed_sequence(self, sequence):
#         data = [("protein1", sequence)]
#         batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
#         with torch.no_grad():
#             results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        
#         token_representations = results["representations"][33]
#         sequence_representation = token_representations[0, 1:len(sequence)+1].mean(0)  # mean-pooled
#         return sequence_representation.numpy()
import torch
import esm

class ProteinEmbedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        # Load the model and alphabet
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        
        # Check if MPS is available, else use CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")  # Fallback to CPU if MPS is not available
        
        # Move model to the selected device (MPS or CPU)
        self.model = self.model.to(self.device)  
        
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disable dropout for deterministic results

    def embed_sequence(self, sequence):
        # Prepare the sequence data
        data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Move the tokens to the selected device (MPS or CPU)
        batch_tokens = batch_tokens.to(self.device)

        # Perform inference on the model
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        # Extract token-level representations and perform mean pooling
        token_representations = results["representations"][33]
        sequence_representation = token_representations[0, 1:len(sequence)+1].mean(0)  # Mean-pooled over sequence length
        
        # Move the result back to CPU and return as a numpy array
        return sequence_representation.cpu().numpy()  # Return as numpy array after moving to CPU
