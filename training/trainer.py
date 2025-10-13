import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, epochs=10, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
            for i, data in loop:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                
                loop.set_description(f"Epoch [{epoch+1}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

        logging.info('Finished Training')
