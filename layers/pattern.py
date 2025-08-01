import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Validator:
    def validate(self, data, labels):
        raise NotImplementedError

class PatternValidator(Validator):
    def __init__(self, input_size, hidden_size=100, output_size=2):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.fitted = False

    def fit(self, X_train, y_train, epochs=200, batch_size=32):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.fitted = True

    def validate(self, data, labels):
        if not self.fitted:
            raise ValueError("Model not fitted")
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            logits = self.model(data)
            probs = nn.Softmax(dim=1)(logits).numpy()
            preds = np.argmax(probs, axis=1)
            accuracy = (preds == labels).mean()
        return preds, probs, accuracy