import torch
import torch.nn as nn

# class LSTM_IC50(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3):
#         super(LSTM_IC50, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):  
#         out, _ = self.lstm(x)
#         out = out[:, -1, :] 
#         return self.regressor(out)

class LSTM_IC50(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, architecture=[64], dropout=0.3, num_layers=1, model_type='lstm'):
        super(LSTM_IC50, self).__init__()
        self.model_type = model_type.lower()

        rnn_class = nn.LSTM if self.model_type in ['lstm', 'bilstm'] else nn.GRU
        bidirectional = self.model_type == 'bilstm'

        self.rnn = rnn_class(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Account for bidirectional output size
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        layers = []
        prev_dim = rnn_output_dim
        for hidden in architecture:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time step
        return self.regressor(out)
