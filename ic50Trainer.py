
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import csv
import numpy as np
class IC50Trainer:
    def __init__(self):
        pass


    def init_log(self,file_path, headers):
        self.file_path=file_path
        with open(self.file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_results(self, row):
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
    def preprocess(self,X, y, test_size=0.2, val_size=0.1, random_state=42,scale=False,batch_size=16):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)

        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.train_loader = self.create_dataloader(self.X_train, self.y_train, batch_size)
        self.val_loader = self.create_dataloader(self.X_val, self.y_val, batch_size, shuffle=False)
        self.test_loader = self.create_dataloader(self.X_test, self.y_test, batch_size, shuffle=False)



    def create_dataloader(self,X,y,batch_size=64,shuffle=True):
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader




    def train(self,model, epochs, optimizer, criterion):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in self.train_loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    output = model(xb)
                    loss = criterion(output, yb)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        return model, avg_train_loss, avg_val_loss


    def test(self,model,verbose=True):
        model.eval()
        all_preds = []
        all_labels = []
        all_fingerprints = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
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
            
                pred_log_ic50 = all_preds[i]
                true_log_ic50 = all_labels[i]
                pred_ic50 = np.expm1(pred_log_ic50)
                true_ic50 = np.expm1(true_log_ic50)

                print(f"Sample {i+1}")
            
                print(f"Predicted log(IC50): {pred_log_ic50:.4f}")
                print(f"True log(IC50):      {true_log_ic50:.4f}")
                print(f"Predicted IC50:      {pred_ic50:.2f} nM")
                print(f"True IC50:           {true_ic50:.2f} nM")
                print("-" * 80)

        return rmse, all_preds, all_labels, all_fingerprints

