
# File: Application/ui/two_moons_training_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QMessageBox,
                             QComboBox, QFileDialog, QSplitter, QTabWidget,QProgressBar)
from PyQt5.QtCore import Qt, pyqtSlot
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from threads.two_moons_training_thread import TwoMoonsTrainingThread
from model.soen_model import SOENModel
from model_config_files.two_moons_config import TwoMoonsConfig
from utils.data_loader import load_two_moons_data
from utils.two_moons_utils import prepare_two_moons_data, get_decision_boundary
import torch
import matplotlib.pyplot as plt
from utils.soen_model_utils import load_soen_model, save_soen_model


class TwoMoonsTrainingTab(QWidget):
    def __init__(self, model, train_loader, val_loader, scaler):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.setup_ui()
        
        self.setup_training_thread()
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.current_epoch = 0
        self.current_batch = 0

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.setup_control_buttons(left_layout)
        self.setup_input_fields(left_layout)
        self.setup_activation_function_dropdown(left_layout)
        self.setup_checkboxes(left_layout)
        left_layout.addStretch(1)
        self.splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.setup_monitoring_widgets(right_layout)
        self.splitter.addWidget(right_widget)
        
        self.splitter.setSizes([1, 3])

    def setup_control_buttons(self, layout):
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.toggle_training)
        layout.addWidget(self.train_button)

        self.save_model_button = QPushButton("Save Model")
        self.save_model_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_model_button)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_button)

        self.save_progress_button = QPushButton("Download Training Progress")
        self.save_progress_button.clicked.connect(self.save_training_progress)
        layout.addWidget(self.save_progress_button)

        self.reset_button = QPushButton("Reset Model")
        self.reset_button.clicked.connect(self.reset_model)
        layout.addWidget(self.reset_button)

    def setup_input_fields(self, layout):
        self.setup_spinbox(layout, "Learning Rate:", 0.001, 0.000001, 1.0, 6, self.update_lr)
        self.setup_spinbox(layout, "Train Noise:", 0.01, 0.0, 1.0, 6, self.update_train_noise)
        self.setup_spinbox(layout, "Test Noise:", 0.01, 0.0, 1.0, 6, self.update_test_noise)
        self.setup_spinbox(layout, "Max Iterations:", 30, 1, 1000, 0, self.update_max_iter, is_int=True)
        self.setup_spinbox(layout, "Epochs:", 10, 1, 1000, 0, self.update_epochs, is_int=True)

    def setup_spinbox(self, layout, label, default, min_val, max_val, decimals, slot, is_int=False):
        layout.addWidget(QLabel(label))
        spinbox = QSpinBox() if is_int else QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        if not is_int:
            spinbox.setDecimals(decimals)
        spinbox.setValue(default)
        spinbox.valueChanged.connect(slot)
        layout.addWidget(spinbox)

    def setup_activation_function_dropdown(self, layout):
        layout.addWidget(QLabel("Activation Function:"))
        self.activation_function_combo = QComboBox()
        self.activation_function_combo.addItems([
            "tanh_2d", "relu_2d", "gaussian_mixture", "sigmoid_mixture","NN_dendrite","tanh_1d","relu_1d"
        ])
        self.activation_function_combo.setCurrentText(self.model.config.activation_function)
        self.activation_function_combo.currentTextChanged.connect(self.update_activation_function)
        layout.addWidget(self.activation_function_combo)

    def setup_checkboxes(self, layout):
        layout.addWidget(QLabel("Learnable Parameters:"))
        self.learnable_param_checkboxes = {}
        learnable_params = ["J"]  # Set "J" as the only learnable parameter initially
        for param in ["J", "gamma", "tau", "flux_offset"]:
            checkbox = QCheckBox(param)
            checkbox.setChecked(param in learnable_params)
            checkbox.stateChanged.connect(self.update_learnable_params)
            layout.addWidget(checkbox)
            self.learnable_param_checkboxes[param] = checkbox

    def setup_monitoring_widgets(self, layout):
        self.plot_tabs = QTabWidget()
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.accuracy_plot = pg.PlotWidget(title="Accuracy")
        self.plot_tabs.addTab(self.loss_plot, "Loss")
        self.plot_tabs.addTab(self.accuracy_plot, "Accuracy")
        layout.addWidget(self.plot_tabs)

        self.decision_boundary_figure = Figure(figsize=(5, 5))
        self.decision_boundary_canvas = FigureCanvas(self.decision_boundary_figure)
        layout.addWidget(self.decision_boundary_canvas)

        # Add progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Add labels for current loss and accuracy
        self.current_stats_layout = QHBoxLayout()
        self.current_loss_label = QLabel("Loss: N/A")
        self.current_accuracy_label = QLabel("Accuracy: N/A")
        self.current_stats_layout.addWidget(self.current_loss_label)
        self.current_stats_layout.addWidget(self.current_accuracy_label)
        layout.addLayout(self.current_stats_layout)

        # Add text boxes for training stats and learnable params
        stats_layout = QHBoxLayout()
        self.training_stats = QTextEdit()
        self.training_stats.setReadOnly(True)
        self.learnable_params_stats = QTextEdit()
        self.learnable_params_stats.setReadOnly(True)
        stats_layout.addWidget(self.training_stats)
        stats_layout.addWidget(self.learnable_params_stats)
        layout.addLayout(stats_layout)

    def setup_training_thread(self, epochs=None):
        self.training_thread = TwoMoonsTrainingThread(self.model, self.train_loader, self.val_loader, self.scaler)
        self.training_thread.update_signal.connect(self.update_ui)
        self.update_learnable_params()
        
        # If epochs is provided, set it in the new training thread
        if epochs is not None:
            self.training_thread.epochs = epochs

    @pyqtSlot()
    def toggle_training(self):
        if self.training_thread.isRunning():
            self.training_thread.stop()
            self.train_button.setText("Start Training")
        else:
            self.training_thread.start()
            self.train_button.setText("Stop Training")

    @pyqtSlot(float)
    def update_lr(self, value):
        self.training_thread.lr = value

    @pyqtSlot(float)
    def update_train_noise(self, value):
        self.training_thread.train_noise_std = value

    @pyqtSlot(float)
    def update_test_noise(self, value):
        self.training_thread.test_noise_std = value

    @pyqtSlot(int)
    def update_max_iter(self, value):
        self.training_thread.max_iter = value

    @pyqtSlot(int)
    def update_epochs(self, value):
        self.training_thread.epochs = value

    def update_learnable_params(self):
        learnable_params = [param for param, checkbox in self.learnable_param_checkboxes.items() if checkbox.isChecked()]
        self.training_thread.learnable_params = learnable_params
        self.model.config.learnable_params = learnable_params

        if not learnable_params:
            self.learnable_param_checkboxes["J"].setChecked(True)
            self.training_thread.learnable_params = ["J"]
            self.model.config.learnable_params = ["J"]

    @pyqtSlot(str)
    def update_activation_function(self, function):
        self.training_thread.activation_function = function
        self.model.set_activation_function(function)



    def update_ui(self, data):
        if data:  # Only update if there's actually data
            self.update_plots(data)
            self.update_training_stats(data)
            self.update_learnable_params_stats()
            if 'decision_boundary_data' in data:
                self.update_decision_boundary(data['decision_boundary_data'])
            self.update_progress_bar(data)




    def update_plots(self, data):
        if data.get('train_losses') and data.get('val_losses'):
            self.loss_plot.clear()
            self.loss_plot.plot(data['batches'], data['train_losses'], pen='r', name='Train Loss')
            self.loss_plot.plot(data['batches'], data['val_losses'], pen='b', name='Val Loss')
        
        if data.get('train_accuracies') and data.get('val_accuracies'):
            self.accuracy_plot.clear()
            self.accuracy_plot.plot(data['batches'], data['train_accuracies'], pen='g', name='Train Acc')
            self.accuracy_plot.plot(data['batches'], data['val_accuracies'], pen='y', name='Val Acc')


    def update_training_stats(self, data):
        self.current_epoch = data.get('current_epoch', self.current_epoch)
        self.current_batch = data.get('current_batch', self.current_batch)
        
        stats = f"Epoch: {self.current_epoch}/{self.training_thread.epochs}\n"
        stats += f"Batch: {self.current_batch}/{len(self.train_loader)}\n"
        
        if data.get('train_losses'):
            self.current_loss_label.setText(f"Loss: {data['train_losses'][-1]:.4f}")
        if data.get('train_accuracies'):
            self.current_accuracy_label.setText(f"Accuracy: {data['train_accuracies'][-1]:.4f}")

        self.training_stats.append(stats)
        
        scrollbar = self.training_stats.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_learnable_params_stats(self):
        stats = "Learnable Parameters:\n"
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats += f"\n{name}:"
                stats += f"  Mean: {param.data.mean().item():.4f},"
                stats += f"  Std: {param.data.std().item():.4f},"
                # stats += f"  Min: {param.data.min().item():.4f},"
                # stats += f"  Max: {param.data.max().item():.4f}."
        
        # Update learnable_params_stats text box
        self.learnable_params_stats.setText(stats)
        
        # Preserve scroll position
        scrollbar = self.learnable_params_stats.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress_bar(self, data):
        progress = (data['current_epoch'] - 1) / self.training_thread.epochs * 100 + \
                   data['current_batch'] / len(self.train_loader) / self.training_thread.epochs * 100
        self.progress_bar.setValue(int(progress))

    def update_decision_boundary(self, data):
        self.decision_boundary_figure.clear()
        ax = self.decision_boundary_figure.add_subplot(111)
        ax.contourf(data['xx'], data['yy'], data['predicted'], cmap=plt.cm.RdYlBu, alpha=0.8)
        ax.scatter(data['X'][:, 0], data['X'][:, 1], c=data['y'], cmap=plt.cm.RdYlBu, edgecolors='black')
        ax.set_xlim(data['x_min'], data['x_max'])
        ax.set_ylim(data['y_min'], data['y_max'])
        self.decision_boundary_canvas.draw()

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth)")
        if file_name:
            try:
                loaded_model = load_soen_model(file_name, SOENModel)
                
                self.model = loaded_model
                self.setup_training_thread()
                
                # Reset all training-related data
                self.train_losses = []
                self.train_accuracies = []
                self.val_losses = []
                self.val_accuracies = []
                self.current_epoch = 0
                self.current_batch = 0
                
                # Update UI without trying to access any training data
                self.update_ui_after_load()
                
                QMessageBox.information(self, "Model Loaded", f"Model successfully loaded from {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                print(f"Error details: {str(e)}")


    def update_ui_after_load(self):
        # Update plots with empty data
        self.loss_plot.clear()
        self.accuracy_plot.clear()
        
        # Reset labels
        self.current_loss_label.setText("Loss: N/A")
        self.current_accuracy_label.setText("Accuracy: N/A")
        
        # Clear training stats
        self.training_stats.clear()
        
        # Update learnable params stats
        self.update_learnable_params_stats()
        
        # Clear decision boundary plot
        self.decision_boundary_figure.clear()
        self.decision_boundary_canvas.draw()
        
        # Reset progress bar
        self.progress_bar.setValue(0)



    def save_model(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Models (*.pth)")
        if file_name:
            try:
                save_soen_model(self.model, file_name)
                QMessageBox.information(self, "Model Saved", f"Model successfully saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
                print(f"Error details: {str(e)}")




    def save_training_progress(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Training Progress", "", "CSV Files (*.csv)")
        if file_path:
            self.training_thread.save_progress_data(file_path)
            QMessageBox.information(self, "Success", f"Training progress data saved to {file_path}")

    
    def reset_model(self):
        if self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait()

        # Store the current epochs value
        current_epochs = self.training_thread.epochs

        new_config = TwoMoonsConfig()
        
        # Update the config with current UI values
        new_config.train_noise_std = self.training_thread.train_noise_std
        new_config.test_noise_std = self.training_thread.test_noise_std
        new_config.max_iter = self.training_thread.max_iter
        new_config.activation_function = self.training_thread.activation_function

        new_model = SOENModel(new_config)

        X, y, _ = prepare_two_moons_data()  # We don't need a new scaler
        X = self.scaler.transform(X)  # Use the existing scaler
        train_loader, val_loader = load_two_moons_data(X, y, self.scaler)

        self.model = new_model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Pass the stored epochs value to setup_training_thread
        self.setup_training_thread(epochs=current_epochs)

        self.loss_plot.clear()
        self.accuracy_plot.clear()
        self.decision_boundary_figure.clear()
        self.training_stats.clear()

        self.update_ui({
            'current_epoch': 0,
            'current_batch': 0,
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'batches': [],
            'decision_boundary_data': get_decision_boundary(self.model, X, y)
        })

        QMessageBox.information(self, "Model Reset", "The model has been reset to its initial state.")
