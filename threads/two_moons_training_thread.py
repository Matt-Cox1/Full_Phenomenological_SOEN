
# File: Application/threads/two_moons_training_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.two_moons_utils import evaluate_model, get_decision_boundary

class TwoMoonsTrainingThread(QThread):
    update_signal = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, model, train_loader, val_loader, scaler):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        self.lr = 0.001
        self.train_noise_std = 0.01
        self.test_noise_std = 0.01
        self.max_iter = 30
        self.learnable_params = ["J"]
        self.epochs = 10
        self.activation_function = model.config.activation_function
        
        self.is_running = False
        
        self.progress_df = pd.DataFrame(columns=[
            'epoch', 'batch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
            'learning_rate', 'train_noise', 'test_noise', 'max_iter'
        ])

    def update_model_params(self):
        self.model.train_noise_std = self.train_noise_std
        self.model.test_noise_std = self.test_noise_std
        self.model.config.max_iter = self.max_iter
        self.model.config.learnable_params = self.learnable_params
        self.model.set_activation_function(self.activation_function)

        for name, param in self.model.named_parameters():
            param.requires_grad = name in self.learnable_params

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

    def run(self):
        self.is_running = True
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        batches = []
        
        for epoch in range(self.epochs):
            if not self.is_running:
                break
            
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if not self.is_running:
                    break
                
                self.update_model_params()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                if batch_idx % 10 == 0:
                    train_accuracy = evaluate_model(self.model, self.train_loader.dataset.tensors[0], self.train_loader.dataset.tensors[1])
                    val_loss, val_accuracy = self.validate()
                    
                    train_losses.append(loss.item())
                    val_losses.append(val_loss)
                    train_accuracies.append(train_accuracy)
                    val_accuracies.append(val_accuracy)
                    batches.append(epoch * len(self.train_loader) + batch_idx)
                    
                    decision_boundary_data = get_decision_boundary(self.model, self.train_loader.dataset.tensors[0].numpy(), self.train_loader.dataset.tensors[1].numpy())
                
                    self.update_signal.emit({
                        'current_epoch': epoch + 1,
                        'current_batch': batch_idx,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                        'batches': batches,
                        'decision_boundary_data': decision_boundary_data
                    })
                    
                    self.update_progress_df(epoch, batch_idx, loss.item(), train_accuracy, val_loss, val_accuracy)

        self.finished.emit()

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy

    def update_progress_df(self, epoch, batch_idx, train_loss, train_accuracy, val_loss, val_accuracy):
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'batch': [batch_idx],
            'train_loss': [train_loss],
            'train_accuracy': [train_accuracy],
            'val_loss': [val_loss],
            'val_accuracy': [val_accuracy],
            'learning_rate': [self.lr],
            'train_noise': [self.train_noise_std],
            'test_noise': [self.test_noise_std],
            'max_iter': [self.max_iter]
        })
        self.progress_df = pd.concat([self.progress_df, new_row], ignore_index=True)

    def get_progress_data(self):
        return self.progress_df

    def save_progress_data(self, file_path):
        self.progress_df.to_csv(file_path, index=False)

    def stop(self):
        self.is_running = False





