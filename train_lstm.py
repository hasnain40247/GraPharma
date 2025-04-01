import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from grapharma_lstm import LSTM_IC50

# Load data
fingerprints = np.load("./representations/bindingdb_processed_fingerprints.npy", allow_pickle=True)
embeddings = np.load("./representations/bindingdb_processed_embeddings.npy", allow_pickle=True)
ic50 = np.load("./representations/ic50.npy")

# Convert to arrays
fingerprints = np.array(fingerprints.tolist())
embeddings = np.array(embeddings.tolist())
ic50 = np.log1p(ic50).reshape(-1, 1).astype(np.float32)

# Pad protein embeddings to match fingerprint size (2048)
if embeddings.shape[1] < fingerprints.shape[1]:
    pad_width = fingerprints.shape[1] - embeddings.shape[1]
    embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), 'constant')

# Stack as 2-step sequences: (batch, seq_len=2, features)
X = np.stack([fingerprints, embeddings], axis=1)
y = ic50

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
model = LSTM_IC50(input_dim=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
def train(model, loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader):.4f}")

train(model, train_loader, epochs=50)

# Testing
def test(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            preds.append(pred.numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds).squeeze()
    trues = np.concatenate(trues).squeeze()
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    print(f"\nTest RMSE (log IC50): {rmse:.4f}")
    return preds, trues

preds, trues = test(model, test_loader)

print("Predictions on Test Data:\n")
for i in range(10):
    pred_log_ic50 = preds[i]
    true_log_ic50 = trues[i]
    pred_ic50 = np.expm1(pred_log_ic50)
    true_ic50 = np.expm1(true_log_ic50)

    print(f"Sample {i+1}")
    print(f"Predicted log(IC50): {pred_log_ic50:.4f}")
    print(f"True log(IC50):      {true_log_ic50:.4f}")
    print(f"Predicted IC50:      {pred_ic50:.2f} nM")
    print(f"True IC50:           {true_ic50:.2f} nM")
    print("-" * 80)

