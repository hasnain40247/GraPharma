

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grapharma_nn import GraPharmaNN



def preprocess(X,y,test_size=0.2,random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def create_dataloader(X,y,batch_size=64,shuffle=True):
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train(model,data,epochs,optimizer,criterion):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in data:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model


def test(model,data,verbose=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_fingerprints = []

    with torch.no_grad():
        for xb, yb in data:
            outputs = model(xb)
            all_preds.extend(outputs.numpy())
            all_labels.extend(yb.numpy())
            all_fingerprints.extend(xb.numpy())


    all_preds = np.array(all_preds).squeeze()
    all_labels = np.array(all_labels).squeeze()
    all_fingerprints = np.array(all_fingerprints)

    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))

    if verbose:
        print(f"\nTest RMSE (log IC50): {rmse:.4f}\n")
        print("Predictions on Test Data:\n")
        for i in range(len(all_preds[:10])):
            fingerprint = all_fingerprints[i]
            pred_log_ic50 = all_preds[i]
            true_log_ic50 = all_labels[i]
            pred_ic50 = np.expm1(pred_log_ic50)
            true_ic50 = np.expm1(true_log_ic50)

            print(f"Sample {i+1}")
            print(f"Fingerprint Vector: {fingerprint}")
            print(f"Predicted log(IC50): {pred_log_ic50:.4f}")
            print(f"True log(IC50):      {true_log_ic50:.4f}")
            print(f"Predicted IC50:      {pred_ic50:.2f} nM")
            print(f"True IC50:           {true_ic50:.2f} nM")
            print("-" * 80)

    return rmse, all_preds, all_labels, all_fingerprints

if __name__ == "__main__":

    fingerprint_file="./representations/bindingdb_processed_fingerprints.npy"
    ic50_file="./representations/ic50.npy"
    epochs=50
    lr=1e-3
    batch_size=64



    X = np.load(fingerprint_file, allow_pickle=True)
    y = np.load(ic50_file)


    X = np.array(X.tolist())  
    y = np.log1p(y).reshape(-1, 1).astype(np.float32) 

    X_train, X_test, y_train, y_test=preprocess(X,y)

    train_loader=create_dataloader(X_train, y_train)
    test_loader=create_dataloader(X_test, y_test)



    model = GraPharmaNN(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    trained_model=train(model,train_loader,epochs,optimizer,criterion)

 
    rmse, all_preds, all_labels, all_fingerprints=test(trained_model,test_loader)

