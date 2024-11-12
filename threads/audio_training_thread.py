# Filename: threads/audio_training_thread.py

"""
This file implements the AudioTrainingThread class, which is responsible for managing the training process
of a neural network model on audio data in a separate thread. It allows for asynchronous training that doesn't block the main GUI thread.
"""

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from enum import Enum
import numpy as np

class TrainingState(Enum):
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2

class AudioTrainingThread(QThread):
    update_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    no_params_signal = pyqtSignal()
    log_signal = pyqtSignal(str)
    state_changed = pyqtSignal(TrainingState)

    def __init__(self, model, train_loader, val_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self._state = TrainingState.STOPPED
        self.lr = 0.001
        self.train_noise_std = 0.00
        self.test_noise_std = 0.00
        self.max_iter = 100
        self.learnable_params = ["J", "tau", "gamma", "flux_offset"]
        self.epochs = 10
        self.current_epoch = 0
        self.current_batch = 0
        self.activation_function = model.config.activation_function

        self.show_weight_matrix = True
        self.show_state_evolution = True

        # Initialise DataFrame for storing training progress
        self.progress_df = pd.DataFrame(columns=[
            'epoch', 'batch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
            'learning_rate', 'train_noise', 'test_noise', 'max_iter'
        ])

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        self.state_changed.emit(state)

    def run(self):
        self.log_signal.emit("Audio training thread started running")
        self.set_state(TrainingState.RUNNING)
        try:
            while self.current_epoch < self.epochs and self._state != TrainingState.STOPPED:
                self.train_epoch()
                self.current_epoch += 1
                if self._state == TrainingState.PAUSED:
                    self.msleep(100)  # Sleep for 100ms when paused
                    continue
                
        except Exception as e:
            self.log_signal.emit(f"Error in audio training thread: {str(e)}")
            self.error_signal.emit(str(e))
        finally:
            self.set_state(TrainingState.STOPPED)
        self.log_signal.emit("Audio training thread finished")

    def wait_while_paused(self):
        while self._state == TrainingState.PAUSED:
            self.msleep(100)  # Sleep for 100ms when paused

    def pause(self):
        self.set_state(TrainingState.PAUSED)

    def resume(self):
        self.set_state(TrainingState.RUNNING)

    def stop(self):
        self.set_state(TrainingState.STOPPED)

    def train_epoch(self):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.model.device)
            target = target.to(self.model.device)
            if self._state == TrainingState.PAUSED:
                self.wait_while_paused()
            if self._state == TrainingState.STOPPED:
                return

            self.model.train()
            self.current_batch = batch_idx
            self.update_model_params()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = output.max(1)
            correct = predicted.eq(target).sum().item()
            total = target.size(0)
            accuracy = 100. * correct / total

            if batch_idx % 10 == 0:  # Emit update every 10 batches
                val_loss, val_accuracy = self.validate()
                self.emit_update(self.current_epoch, batch_idx, loss.item(), accuracy, val_loss, val_accuracy)

        self.scheduler.step()

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.val_loader):
                data = data.to(self.model.device)
                target = target.to(self.model.device)
                if i >= 10:  # Limit validation to 10 batches for speed
                    break
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= min(10, len(self.val_loader))
        val_accuracy = 100. * correct / total
        return val_loss, val_accuracy

    def update_model_params(self):
        self.model.train_noise_std = self.train_noise_std
        self.model.test_noise_std = self.test_noise_std
        self.model.config.max_iter = self.max_iter
        self.model.config.learnable_params = self.learnable_params
        self.model.set_activation_function(self.activation_function)

        self.log_signal.emit(f"Updated model parameters: "
                             f"train_noise={self.train_noise_std}, "
                             f"test_noise={self.test_noise_std}, "
                             f"max_iter={self.max_iter}, "
                             f"activation={self.activation_function}")

        for name, param in self.model.named_parameters():
            param.requires_grad = name in self.learnable_params

        if not any(p.requires_grad for p in self.model.parameters()):
            self.no_params_signal.emit()
            self.is_paused = True
            return

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

    def emit_update(self, epoch, batch_idx, loss, accuracy, val_loss, val_accuracy):
        weight_matrix = self.model.J.detach().cpu().numpy() if self.show_weight_matrix else None
        
        if self.show_state_evolution and self.model.state_evolution:
            final_states = self.model.state_evolution[-1].detach().cpu().numpy()
            sample_idx = np.random.randint(final_states.shape[0])
            input_states = final_states[sample_idx, :self.model.num_input]
            hidden_states = final_states[sample_idx, self.model.num_input:-self.model.num_output]
            output_states = final_states[sample_idx, -self.model.num_output:]
            final_state_info = {
                'input_states': input_states,
                'hidden_states': hidden_states,
                'output_states': output_states,
                'num_input': self.model.num_input,
                'num_hidden': self.model.num_hidden,
                'num_output': self.model.num_output
            }
        else:
            final_state_info = None

        update_data = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': len(self.train_loader),
            'loss': loss,
            'accuracy': accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'weight_matrix': weight_matrix,
            'final_state_info': final_state_info,
            'text_info': self.get_text_info(epoch, batch_idx, loss, accuracy, val_loss, val_accuracy),
            'param_stats_info': self.get_param_stats_info(),
            'is_validation': False
        }

        self.update_signal.emit(update_data)
        
        # Emit a separate update for validation data
        if val_loss is not None and val_accuracy is not None:
            val_update_data = update_data.copy()
            val_update_data['is_validation'] = True
            val_update_data['loss'] = val_loss
            val_update_data['accuracy'] = val_accuracy
            self.update_signal.emit(val_update_data)
        
        # Add data to progress DataFrame
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'batch': [batch_idx],
            'train_loss': [loss],
            'train_accuracy': [accuracy],
            'val_loss': [val_loss],
            'val_accuracy': [val_accuracy],
            'learning_rate': [self.lr],
            'train_noise': [self.train_noise_std],
            'test_noise': [self.test_noise_std],
            'max_iter': [self.max_iter]
        })
        self.progress_df = pd.concat([self.progress_df, new_row], ignore_index=True)

    def get_text_info(self, epoch, batch_idx, loss, accuracy, val_loss, val_accuracy):
        return (f"Training - Epoch: {epoch+1}/{self.epochs}, "
                f"Batch: {batch_idx}/{len(self.train_loader)}\n"
                f"Train Loss: {loss:.6f}, Train Accuracy: {accuracy:.2f}%\n"
                f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%\n"
                f"Learning Rate: {self.lr:.6f}\n"
                f"Train Noise: {self.train_noise_std:.6f}, Test Noise: {self.test_noise_std:.6f}\n"
                f"Max Iterations: {self.max_iter}\n"
                f"Learnable Parameters: {', '.join(self.learnable_params)}")

    def get_param_stats_info(self):
        stats = []
        for name, param in self.model.named_parameters():
            stats.append(f"{name} - Mean: {param.mean().item():.6f}, Std: {param.std().item():.6f}")
        return "Parameter Statistics:\n" + "\n".join(stats)

    def get_progress_data(self):
        return self.progress_df

    def save_progress_data(self, file_path):
        self.progress_df.to_csv(file_path, index=False)
        self.log_signal.emit(f"Training progress data saved to {file_path}")