import torch.nn as nn
class GraPharmaNN(nn.Module):
    def __init__(self, input_dim, architecture=[512, 128], dropout=0.2):
        super(GraPharmaNN, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in architecture:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
