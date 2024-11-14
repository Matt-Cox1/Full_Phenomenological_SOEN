# FILENAME: ui/spoken_digit_tab.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QMessageBox,
                              QFileDialog,QComboBox,QSplitter,QTabWidget,QPlainTextEdit)
              
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt, pyqtSlot
import pyqtgraph as pg
from threads.audio_training_thread import AudioTrainingThread, TrainingState  
from config import (DEFAULT_LEARNING_RATE, DEFAULT_TRAIN_NOISE, DEFAULT_TEST_NOISE, 
                    DEFAULT_MAX_ITER, PLOT_WIDTH, PLOT_HEIGHT)
import logging
from utils.plotting import setup_plot, update_line_plot, update_image_plot
import torch
from utils.soen_model_utils import load_soen_model, save_soen_model
from model_config_files.spokendigit_rnn_config import SpokenDigitRNNConfig
from model.soen_model import SOENModel
from utils.data_loader import load_audio_data
import numpy as np
from utils.network_analysis import analyse_network

"""
This file defines the TrainingTab class, a central component of the SOEN Model Monitor application.
It provides a user interface for training SOEN models on the MNIST dataset and visualising the training process.
"""

def format_section(title, content):
    """Helper function to format sections of the network analysis report."""
    separator = "=" * 80
    return f"\n{title}\n{separator}\n{content}\n\n"

class SpokenDigitTrainingTab(QWidget):
    def __init__(self, model, train_loader, val_loader):
        super().__init__()
        self.model = model
        
        # Use the provided loaders instead of loading new ones
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Print data shapes
        train_data = next(iter(self.train_loader))[0]  # Get first batch of training data
        val_data = next(iter(self.val_loader))[0]      # Get first batch of validation data
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Validation data shape: {val_data.shape}")
        
        self.learnable_param_checkboxes = {}
        self.show_weight_matrix = True
        self.show_state_evolution = True
        self.setup_ui()
        self.setup_training_thread()
        self.reset_plots()
        self.perform_network_analysis()

        


    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Create main horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left column for controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.setup_control_buttons(left_layout)
        self.setup_input_fields(left_layout)
        self.setup_checkboxes(left_layout)
        self.setup_activation_function_dropdown(left_layout)
        left_layout.addStretch(1)
        self.main_splitter.addWidget(left_widget)
        
        # Right column for plots and info
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Create vertical splitter for plots and info
        self.plot_info_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(self.plot_info_splitter)
        
        # Add plots
        self.plot_tabs = QTabWidget()
        self.setup_plots(self.plot_tabs)
        self.plot_info_splitter.addWidget(self.plot_tabs)
        
        # Add info areas to a horizontal splitter
        info_splitter = QSplitter(Qt.Horizontal)
        self.setup_info_areas(info_splitter)
        self.plot_info_splitter.addWidget(info_splitter)
        
        self.main_splitter.addWidget(right_widget)
        
        # Set initial sizes
        self.main_splitter.setSizes([1, 3])  # Left column takes 1/4, right column takes 3/4
        self.plot_info_splitter.setSizes([2, 1])  # Plots take 2/3, info areas take 1/3

    def toggle_weight_matrix(self, state):
        self.show_weight_matrix = bool(state)
        self.weight_plot.setVisible(self.show_weight_matrix)

    def toggle_state_evolution(self, state):
        self.show_state_evolution = bool(state)
        self.input_state_plot.setVisible(self.show_state_evolution)
        self.hidden_state_plot.setVisible(self.show_state_evolution)
        self.output_state_plot.setVisible(self.show_state_evolution)

    def setup_control_buttons(self, layout):
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.toggle_training)
        layout.addWidget(self.train_button)
        
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_button)

        self.download_progress_button = QPushButton("Download Training Progress")
        self.download_progress_button.clicked.connect(self.download_training_progress)
        layout.addWidget(self.download_progress_button)

        self.reset_button = QPushButton("Reset Model")
        self.reset_button.clicked.connect(self.reset_model)
        layout.addWidget(self.reset_button)


    def reset_model(self):
        if self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait()

        # Store current training thread settings
        current_settings = {
            'epochs': self.training_thread.epochs,
            'lr': self.training_thread.lr,
            'train_noise_std': self.training_thread.train_noise_std,
            'test_noise_std': self.training_thread.test_noise_std,
            'max_iter': self.training_thread.max_iter,
            'learnable_params': self.training_thread.learnable_params,
            'activation_function': self.training_thread.activation_function
        }

        # Create a new model with the original configuration
        new_config = SpokenDigitRNNConfig()
        new_model = SOENModel(new_config)

        # Load the preprocessed data directly
        self.train_loader, self.val_loader, _ = load_audio_data()

        self.model = new_model
        
        # Set up the new training thread with preserved settings
        self.setup_training_thread()
        
        # Update the new training thread with the preserved settings
        self.training_thread.epochs = current_settings['epochs']
        self.training_thread.lr = current_settings['lr']
        self.training_thread.train_noise_std = current_settings['train_noise_std']
        self.training_thread.test_noise_std = current_settings['test_noise_std']
        self.training_thread.max_iter = current_settings['max_iter']
        self.training_thread.learnable_params = current_settings['learnable_params']
        self.training_thread.activation_function = current_settings['activation_function']

        # Update UI elements to reflect the preserved settings
        self.update_ui_elements(current_settings)

        self.reset_plots()
        self.update_plots({})

        self.perform_network_analysis()

        QMessageBox.information(self, "Model Reset", "The model has been reset to its initial state while preserving training settings.")


    def update_ui_elements(self, settings):
        # Update UI elements to reflect the preserved settings
        for widget in self.findChildren((QDoubleSpinBox, QSpinBox)):
            if widget.objectName() == 'learning_rate_spinbox':
                widget.setValue(settings['lr'])
            elif widget.objectName() == 'train_noise_spinbox':
                widget.setValue(settings['train_noise_std'])
            elif widget.objectName() == 'test_noise_spinbox':
                widget.setValue(settings['test_noise_std'])
            elif widget.objectName() == 'max_iter_spinbox':
                widget.setValue(settings['max_iter'])
            elif widget.objectName() == 'epochs_spinbox':
                widget.setValue(settings['epochs'])

        # Update activation function dropdown
        if hasattr(self, 'activation_function_dropdown'):
            index = self.activation_function_dropdown.findText(settings['activation_function'])
            if index >= 0:
                self.activation_function_dropdown.setCurrentIndex(index)

        # Update learnable parameters checkboxes
        for param, checkbox in self.learnable_param_checkboxes.items():
            checkbox.setChecked(param in settings['learnable_params'])
        

    def reset_plots(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.batches = []
        self.val_batches = []
        self.plot_update_frequency = 50  # Update plots every 50 batches



    def setup_input_fields(self, layout):
        self.setup_spinbox(layout, "Learning Rate:", DEFAULT_LEARNING_RATE, 0.00000, 10.0, 6, self.update_lr)
        self.setup_spinbox(layout, "Train Noise:", DEFAULT_TRAIN_NOISE, 0.0, 10.0, 6, self.update_train_noise)
        self.setup_spinbox(layout, "Test Noise:", DEFAULT_TEST_NOISE, 0.0, 10.0, 6, self.update_test_noise)
        self.setup_spinbox(layout, "Max Iterations:", DEFAULT_MAX_ITER, 1, 1000, 0, self.update_max_iter, is_int=True)
        self.setup_spinbox(layout, "Epochs:", 10, 1, 1000, 0, self.update_epochs, is_int=True)
        
    def download_training_progress(self):
        if hasattr(self, 'training_thread'):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Training Progress Data", "", "CSV Files (*.csv)")
            if file_path:
                self.training_thread.save_progress_data(file_path)
                QMessageBox.information(self, "Success", f"Training progress data saved to {file_path}")
        else:
            QMessageBox.warning(self, "Warning", "No training data available. Please start training first.")



    def setup_activation_function_dropdown(self, layout):
        layout.addWidget(QLabel("Activation Function:"))
        self.activation_function_dropdown = QComboBox()
        self.activation_function_dropdown.addItems([
            "tanh_2d", "relu_2d", "gaussian_mixture", "sigmoid_mixture","NN_dendrite","tanh_1d","relu_1d"
        ])
        self.activation_function_dropdown.setCurrentText(self.model.config.activation_function)
        self.activation_function_dropdown.currentTextChanged.connect(self.update_activation_function)
        layout.addWidget(self.activation_function_dropdown)
    

    def setup_spinbox(self, layout, label, default, min_val, max_val, decimals, slot, is_int=False):
        layout.addWidget(QLabel(label))
        spinbox = QSpinBox() if is_int else QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        if not is_int:
            spinbox.setDecimals(decimals)
        spinbox.setValue(default)
        spinbox.valueChanged.connect(slot)
        layout.addWidget(spinbox)

    def setup_checkboxes(self, layout):
        layout.addWidget(QLabel("Learnable Parameters:"))
        for param in ["J", "gamma", "tau", "flux_offset"]:
            checkbox = QCheckBox(param)
            checkbox.setChecked(param in self.model.config.learnable_params) 
            checkbox.stateChanged.connect(self.update_learnable_params)
            layout.addWidget(checkbox)
            self.learnable_param_checkboxes[param] = checkbox


        layout.addWidget(QLabel("Visualization Options:"))
        self.weight_matrix_toggle = QCheckBox("Show Weight Matrix")
        self.weight_matrix_toggle.setChecked(True)
        self.weight_matrix_toggle.stateChanged.connect(self.toggle_weight_matrix)
        layout.addWidget(self.weight_matrix_toggle)

        self.state_evolution_toggle = QCheckBox("Show State Values")
        self.state_evolution_toggle.setChecked(True)
        self.state_evolution_toggle.stateChanged.connect(self.toggle_state_evolution)
        layout.addWidget(self.state_evolution_toggle)

    def toggle_weight_matrix(self, state):
        self.show_weight_matrix = bool(state)
        self.weight_plot.setVisible(self.show_weight_matrix)

    def toggle_state_evolution(self, state):
        self.show_state_evolution = bool(state)
        self.input_state_plot.setVisible(self.show_state_evolution)
        self.hidden_state_plot.setVisible(self.show_state_evolution)
        self.output_state_plot.setVisible(self.show_state_evolution)

    
    def setup_plots(self, tab_widget):
        # Tab for Loss and Accuracy plots
        loss_acc_tab = QWidget()
        loss_acc_layout = QHBoxLayout(loss_acc_tab)
        self.loss_plot = setup_plot("Loss", PLOT_WIDTH, PLOT_HEIGHT)
        self.acc_plot = setup_plot("Accuracy", PLOT_WIDTH, PLOT_HEIGHT)
        loss_acc_layout.addWidget(self.loss_plot)
        loss_acc_layout.addWidget(self.acc_plot)
        tab_widget.addTab(loss_acc_tab, "Loss & Accuracy")

        # Tab for Audio and State Visualization
        audio_state_tab = QWidget()
        audio_state_layout = QVBoxLayout(audio_state_tab)

        # Top row: Mel Spectrogram and Network Response
        top_row = QSplitter(Qt.Horizontal)
        self.mel_spec_plot = pg.ImageView()
        self.mel_spec_plot.ui.histogram.hide()
        self.mel_spec_plot.ui.roiBtn.hide()
        self.mel_spec_plot.ui.menuBtn.hide()
        self.network_response_plot = pg.ImageView()
        self.network_response_plot.ui.histogram.hide()
        self.network_response_plot.ui.roiBtn.hide()
        self.network_response_plot.ui.menuBtn.hide()
        top_row.addWidget(self.mel_spec_plot)
        top_row.addWidget(self.network_response_plot)
        audio_state_layout.addWidget(top_row)

        # Bottom row: Hidden State Evolution
        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        self.hidden_state_plot = pg.ImageView()
        self.hidden_state_plot.ui.histogram.hide()
        self.hidden_state_plot.ui.roiBtn.hide()
        self.hidden_state_plot.ui.menuBtn.hide()
        bottom_layout.addWidget(self.hidden_state_plot)
        audio_state_layout.addWidget(bottom_row)

        tab_widget.addTab(audio_state_tab, "Audio & States")

        # Tab for Network Analysis
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.network_analysis_text = QPlainTextEdit()
        self.network_analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.network_analysis_text)
        tab_widget.addTab(analysis_tab, "Network Analysis")

        # Set equal sizes for rows in the audio state tab
        audio_state_layout.setStretch(0, 1)
        audio_state_layout.setStretch(1, 1)

        # Initial network analysis
        self.update_network_analysis()

        

    

    def setup_info_areas(self, layout):
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        self.param_stats_info = QTextEdit()
        self.param_stats_info.setReadOnly(True)
        layout.addWidget(self.text_info)
        layout.addWidget(self.param_stats_info)

    @pyqtSlot(int)
    def update_epochs(self, value):
        self.training_thread.epochs = value

    


    def update_plots(self, data):
        if not data:  # Reset plots
            self.loss_plot.clear()
            self.acc_plot.clear()
            self.mel_spec_plot.clear()
            self.network_response_plot.clear()
            self.hidden_state_plot.clear()
            return

        is_validation = data.get('is_validation', False)
        
        if is_validation:
            self.val_losses.append(data['loss'])
            self.val_accs.append(data['accuracy'])
            self.val_batches.append(data['epoch'] * len(self.train_loader) + data['batch'])
        else:
            self.batches.append(data['epoch'] * len(self.train_loader) + data['batch'])
            self.train_losses.append(data['loss'])
            self.train_accs.append(data['accuracy'])

        # Update plots every plot_update_frequency batches or on validation data
        if len(self.batches) % self.plot_update_frequency == 0 or is_validation:
            # Update training plots
            update_line_plot(self.loss_plot, self.batches, self.train_losses, "Train Loss", 'r')
            update_line_plot(self.acc_plot, self.batches, self.train_accs, "Train Acc", 'r')
        
            # Update validation plots
            if self.val_losses:
                update_line_plot(self.loss_plot, self.val_batches, self.val_losses, "Val Loss", 'b')
                update_line_plot(self.acc_plot, self.val_batches, self.val_accs, "Val Acc", 'b')
        
            if 'mel_spectrogram' in data and data['mel_spectrogram'] is not None:
                self.mel_spec_plot.setImage(data['mel_spectrogram'].T)
                self.mel_spec_plot.setTitle("Mel Spectrogram")

            if 'network_response' in data and data['network_response'] is not None:
                self.network_response_plot.setImage(data['network_response'].T)
                self.network_response_plot.setTitle("Network Response")

            if 'hidden_states' in data and data['hidden_states'] is not None:
                self.hidden_state_plot.setImage(data['hidden_states'].T)
                self.hidden_state_plot.setTitle("Hidden State Evolution")

        self.text_info.setText(data.get('text_info', ''))
        self.param_stats_info.setText(data.get('param_stats_info', ''))
        



    def toggle_pause(self):
        self.training_thread.paused = not self.training_thread.paused
        self.pause_button.setText("Resume" if self.training_thread.paused else "Pause")

    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Models (*.pth)")
        if file_path:
            try:
                save_soen_model(self.model, file_path)
                QMessageBox.information(self, "Model Saved", f"Model successfully saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")


    @pyqtSlot()
    def update_learnable_params(self):
        learnable_params = [param for param, checkbox in self.learnable_param_checkboxes.items() if checkbox.isChecked()]
        self.training_thread.learnable_params = learnable_params
        self.model.config.learnable_params = learnable_params

        # If no parameters are selected, automatically check J
        if not learnable_params:
            self.learnable_param_checkboxes["J"].setChecked(True)
            self.training_thread.learnable_params = ["J"]
            self.model.config.learnable_params = ["J"]

    @pyqtSlot()
    def handle_no_params(self):
        logging.warning("No learnable parameters selected")
        QMessageBox.warning(self, "Warning", "No learnable parameters selected. J will be set as default.",
                            QMessageBox.Ok)
        self.J_checkbox.setChecked(True)
        self.update_learnable_params()
        if self.training_thread.paused:
            self.toggle_pause()  # Resume training

    def show_error_message(self, message):
        logging.error(f"Training error: {message}")
        QMessageBox.critical(self, "Error", f"An error occurred during training:\n{message}",
                             QMessageBox.Ok)
        self.train_button.setText("Resume Training")
        self.training_thread.pause()

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth)")
        if file_path:
            try:
                loaded_model = load_soen_model(file_path, SOENModel)

                # Update the model
                self.model = loaded_model
                logging.info(f"Model loaded from {file_path}")

                # Reset the training thread with the new model
                self.reset_training_thread()
                logging.info("Training thread reset with new model")

                # Update UI elements to reflect the new model's configuration
                self.update_ui_for_loaded_model()

                # Perform network analysis for the new model
                self.perform_network_analysis()

                QMessageBox.information(self, "Model Loaded", f"Model successfully loaded from {file_path}")
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    


    def toggle_training(self):
        if self.training_thread.get_state() == TrainingState.STOPPED:
            self.start_training()
        elif self.training_thread.get_state() == TrainingState.RUNNING:
            self.pause_training()
        elif self.training_thread.get_state() == TrainingState.PAUSED:
            self.resume_training()

    def start_training(self):
        self.training_thread.start()
        self.train_button.setText("Pause Training")

    def pause_training(self):
        self.training_thread.pause()
        self.train_button.setText("Resume Training")

    def resume_training(self):
        self.training_thread.resume()
        self.train_button.setText("Pause Training")

    @pyqtSlot(TrainingState)
    def on_training_state_changed(self, state):
        if state == TrainingState.STOPPED:
            self.train_button.setText("Start Training")
        elif state == TrainingState.RUNNING:
            self.train_button.setText("Pause Training")
        elif state == TrainingState.PAUSED:
            self.train_button.setText("Resume Training")

    def update_ui_for_loaded_model(self):
        config = self.model.config
        
        # Update activation function dropdown
        if hasattr(self, 'activation_function_dropdown'):
            self.activation_function_dropdown.setCurrentText(config.activation_function)
        
        # Update other UI elements based on the loaded model's configuration
        for widget in self.findChildren((QDoubleSpinBox, QSpinBox)):
            if isinstance(widget, QDoubleSpinBox):
                if widget.objectName() == 'learning_rate':
                    widget.setValue(self.training_thread.lr)
                elif widget.objectName() == 'train_noise':
                    widget.setValue(config.train_noise_std)
                elif widget.objectName() == 'test_noise':
                    widget.setValue(config.test_noise_std)
            elif isinstance(widget, QSpinBox):
                if widget.objectName() == 'max_iter':
                    widget.setValue(config.max_iter)
                elif widget.objectName() == 'epochs':
                    widget.setValue(self.training_thread.epochs)

        # Update learnable parameters checkboxes
        for param, checkbox in self.learnable_param_checkboxes.items():
            checkbox.setChecked(param in config.learnable_params)

        # Log the update
        logging.info("UI updated for loaded model")



    def setup_training_thread(self, epochs=None):
        self.training_thread = AudioTrainingThread(self.model, self.train_loader, self.val_loader)
        self.training_thread.update_signal.connect(self.update_plots)
        self.training_thread.error_signal.connect(self.show_error_message)
        self.training_thread.no_params_signal.connect(self.handle_no_params)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.state_changed.connect(self.on_training_state_changed)
        
        # If epochs is provided, set it in the new training thread
        if epochs is not None:
            self.training_thread.epochs = epochs
    
    @pyqtSlot(str)
    def log_message(self, message):
        self.text_info.append(message)

    
    @pyqtSlot()
    def on_training_finished(self):
        self.train_button.setText("Start Training")
        logging.info("Training finished")

    def reset_training_thread(self):
        if hasattr(self, 'training_thread'):
            self.training_thread.stop()
            self.training_thread.wait()
        self.setup_training_thread()
        self.train_button.setText("Start Training")



    @pyqtSlot(float)
    def update_lr(self, value):
        self.training_thread.lr = value
        logging.info(f"Learning rate updated to: {value}")

    @pyqtSlot(float)
    def update_train_noise(self, value):
        self.training_thread.train_noise_std = value
        self.model.train_noise_std = value
        logging.info(f"Train noise updated to: {value}")

    @pyqtSlot(float)
    def update_test_noise(self, value):
        self.training_thread.test_noise_std = value
        self.model.test_noise_std = value
        logging.info(f"Test noise updated to: {value}")

    @pyqtSlot(int)
    def update_max_iter(self, value):
        self.training_thread.max_iter = value
        self.model.max_iter = value
        logging.info(f"Max iterations updated to: {value}")

    def update_activation_function(self, new_function):
        self.training_thread.activation_function = new_function
        logging.info(f"Activation function updated to: {new_function}")







    ##########################################
    #              NETWORK ANALYSIS          #
    ##########################################
    def perform_network_analysis(self):
        analysis = analyse_network(self.model, verbosity_level=3)
        analysis_text = self.format_network_analysis(analysis)
        if self.network_analysis_text:
            self.network_analysis_text.setPlainText(analysis_text)

    def update_network_analysis(self):
        analysis = analyse_network(self.model, verbosity_level=3)
        analysis_text = self.format_network_analysis(analysis)
        self.network_analysis_text.setPlainText(analysis_text)

    def format_network_analysis(self, analysis):
        def format_param_stats(param_stats):
            content = "Parameter   |   Mean   |   Std Dev\n"
            content += "────────────┼──────────┼───────────\n"
            for param, stats in param_stats.items():
                # Add type checking for mean and std values
                mean = f"{stats['mean']:8.4f}" if isinstance(stats['mean'], (int, float)) else str(stats['mean'])
                std = f"{stats['std']:9.4f}" if isinstance(stats['std'], (int, float)) else str(stats['std'])
                content += f"{param:11} | {mean} | {std}\n"
            return content

        def parse_value(value):
            if isinstance(value, str):
                # Handle K/M suffixes
                if value.endswith('K'):
                    return int(float(value[:-1]) * 1000)
                elif value.endswith('M'):
                    return int(float(value[:-1]) * 1000000)
                # Handle decimal numbers
                try:
                    return int(float(value))
                except ValueError:
                    return 0  # Return 0 if conversion fails
            return int(value) if isinstance(value, (int, float)) else 0

        def format_distribution(distribution_data):
            # Handle case where distribution_data is a list
            if isinstance(distribution_data, list):
                content = "Connections | Count | Distribution\n"
                content += "───────────┼───────┼────────────────────────────────────────────\n"
                for item in distribution_data:
                    connections = item.get('connections', 'N/A')
                    count = item.get('count', 0)
                    bar = item.get('bar', '')
                    # Format the line
                    conn = f"{connections:11}" if isinstance(connections, str) else f"{connections:11d}"
                    count_str = f"{count:5d}" if isinstance(count, (int, float)) else f"{str(count):5}"
                    content += f"{conn} | {count_str} | {bar}\n"
                return content
            
            # Handle case where distribution_data is a dictionary
            if distribution_data.get("empty", False):
                return distribution_data.get("message", "No data available.")
            
            content = "Connections | Count | Distribution\n"
            content += "───────────┼───────┼────────────────────────────────────────────\n"
            for connections, data in distribution_data["distribution"].items():
                bar = data["bar"]
                # Add type checking for count
                count = f"{data['count']:5d}" if isinstance(data['count'], (int, float)) else str(data['count'])
                # Ensure connections is properly formatted
                conn = f"{connections:11d}" if isinstance(connections, (int, float)) else f"{str(connections):11}"
                content += f"{conn} | {count} | {bar}\n"
            return content

        formatted_text = "SOEN Network Analysis\n"
        formatted_text += "=" * 80 + "\n\n"

        if "network_structure" in analysis:
            # Modified formatting to handle different value types
            content = []
            for k, v in analysis["network_structure"].items():
                key = k.replace('_', ' ').title()
                # Format numbers with comma separator, leave other types as-is
                value = f"{v:,}" if isinstance(v, (int, float)) else str(v)
                content.append(f"{key:15} {value}")
            formatted_text += format_section("Network Structure", "\n".join(content))

        if "parameter_statistics" in analysis:
            formatted_text += format_section("Parameter Statistics", format_param_stats(analysis["parameter_statistics"]))

        if "parameters" in analysis:
            content = ""
            for name, info in analysis["parameters"]["params_info"].items():
                content += f"{name:12} Shape {str(info['shape']):15} "
                value = info['non_zero_elements' if name == 'J' else 'elements']
                content += f"{'Non-zero elements' if name == 'J' else 'Elements':20} {parse_value(value):,}\n"
            content += f"\nTotal Parameters:        {parse_value(analysis['parameters']['total_params']):,}\n"
            content += f"Non-zero Mask:           {parse_value(analysis['parameters']['non_zero_mask']):,}\n"
            content += f"Non-zero Weights:        {parse_value(analysis['parameters']['non_zero_weights']):,}"
            formatted_text += format_section("Parameters", content)

        if "weight_matrix" in analysis:
            wm = analysis["weight_matrix"]
            # Add type checking for sparsity
            sparsity = f"{wm['sparsity']*100:.2f}" if isinstance(wm['sparsity'], (int, float)) else str(wm['sparsity'])
            content = f"Shape:                {wm['shape']}\n"
            content += f"Non-zero Elements:    {parse_value(wm['non_zero']):,}\n"
            content += f"Sparsity:             {sparsity}%"
            formatted_text += format_section("Weight Matrix Analysis", content)

        if "overall_statistics" in analysis:
            os = analysis["overall_statistics"]
            # Add type checking for sparsity
            sparsity = f"{os['sparsity']*100:.2f}" if isinstance(os['sparsity'], (int, float)) else str(os['sparsity'])
            content = f"Total Possible Connections:   {parse_value(os['total_possible']):,}\n"
            content += f"Actual Connections:           {parse_value(os['actual_connections']):,}\n"
            content += f"Sparsity:                     {sparsity}%"
            formatted_text += format_section("Overall Statistics", content)

        if "input_node_connections" in analysis:
            formatted_text += format_section("Input Node Connectivity Distribution", 
                                            format_distribution(analysis["input_node_connections"]))

        if "hidden_node_connections" in analysis:
            # Handle multiple hidden layers
            if isinstance(analysis["hidden_node_connections"], list):
                for i, layer_data in enumerate(analysis["hidden_node_connections"]):
                    formatted_text += format_section(f"Hidden Layer {i+1} Connectivity Distribution", 
                                                  format_distribution(layer_data))
            else:
                # Handle single layer case
                formatted_text += format_section("Hidden Node Connectivity Distribution", 
                                              format_distribution(analysis["hidden_node_connections"]))

        if "output_node_connections" in analysis:
            formatted_text += format_section("Output Node Connectivity Distribution", 
                                            format_distribution(analysis["output_node_connections"]))

        return formatted_text










#